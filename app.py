import io
import json
import math
import time
import zlib
import base64
import secrets
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

try:
    import soundfile as sf
except Exception as e:
    sf = None

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization


@dataclass
class AudioData:
    sr: int
    samples: np.ndarray  # float32 in [-1, 1], shape (n,) mono
    orig_format: str


def _to_mono_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)

    mx = np.max(np.abs(x)) if x.size else 1.0
    if mx > 1.0:
        x = x / mx
    return x


def read_audio(file_bytes: bytes, filename: str) -> AudioData:
    ext = filename.lower().split(".")[-1] if "." in filename else "wav"

    if sf is not None and ext in ("wav", "flac", "ogg"):
        data, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
        samples = _to_mono_float32(data)
        return AudioData(sr=sr, samples=samples, orig_format=ext)

    if AudioSegment is None:
        raise RuntimeError("pydub not installed. Install pydub (and ffmpeg for mp3).")

    audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=ext)
    sr = audio.frame_rate
    ch = audio.channels

    arr = np.array(audio.get_array_of_samples())
    if ch > 1:
        arr = arr.reshape((-1, ch)).mean(axis=1).astype(np.int16)

    max_val = float(2 ** (8 * audio.sample_width - 1))
    samples = (arr.astype(np.float32) / max_val).clip(-1, 1)
    return AudioData(sr=sr, samples=samples, orig_format=ext)


def write_wav_bytes(samples_float: np.ndarray, sr: int) -> bytes:
    if sf is None:
        raise RuntimeError("soundfile is required to write WAV. Install soundfile.")
    buf = io.BytesIO()

    pcm = float_to_int16(samples_float)
    sf.write(buf, pcm, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).round().astype(np.int16)


def int16_to_float(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32767.0).clip(-1, 1)


def audio_fingerprint(samples: np.ndarray, sr: int) -> bytes:
    x = samples
    target_sr = 8000
    if resample_poly is not None and sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        x = resample_poly(x, up, down).astype(np.float32)

    mx = np.max(np.abs(x)) if x.size else 1.0
    if mx > 0:
        x = x / mx

    win = 512
    hop = 256
    if x.size < win:
        x = np.pad(x, (0, win - x.size))

    energies = []
    for i in range(0, x.size - win + 1, hop):
        seg = x[i:i + win]
        e = float(np.mean(seg * seg))
        energies.append(e)

    energies = np.array(energies, dtype=np.float32)

    energies = np.log10(energies + 1e-9)
    q = np.clip(((energies - energies.min()) / (np.ptp(energies) + 1e-9)) * 255.0, 0, 255).astype(np.uint8)
    raw = q.tobytes()


    comp = zlib.compress(raw, level=9)
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    return crc.to_bytes(4, "big") + comp


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    d = a[:n] - b[:n]
    return float(np.mean(d * d))


def snr_db(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(a.size, b.size)
    if n == 0:
        return float("inf")
    noise = a[:n] - b[:n]
    ps = float(np.mean(a[:n] * a[:n]))
    pn = float(np.mean(noise * noise))
    return 10.0 * math.log10((ps + 1e-12) / (pn + 1e-12))


def psnr_db(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    m = mse(a, b)
    return 10.0 * math.log10((peak * peak + 1e-12) / (m + 1e-12))


def generate_keypair() -> Tuple[bytes, bytes]:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_bytes, pub_bytes


def load_private_key(pem: bytes) -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(pem, password=None)


def load_public_key(pem: bytes) -> Ed25519PublicKey:
    return serialization.load_pem_public_key(pem)


def canonical_json(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_payload_container(
    meta: Dict[str, Any],
    fingerprint_bytes: bytes,
    alg_version: int,
) -> Dict[str, Any]:
    return {
        "alg_version": alg_version,
        "ts_utc": int(time.time()),
        "id": secrets.token_hex(16),  # 32 hex chars
        "meta": meta,
        "fp_b64": base64.b64encode(fingerprint_bytes).decode("ascii"),
    }


def sign_container(priv_pem: bytes, container: Dict[str, Any]) -> Dict[str, Any]:
    priv = load_private_key(priv_pem)
    data = canonical_json(container)
    sig = priv.sign(data)
    out = dict(container)
    out["sig_b64"] = base64.b64encode(sig).decode("ascii")
    return out


def verify_container(pub_pem: bytes, container_with_sig: Dict[str, Any]) -> bool:
    pub = load_public_key(pub_pem)
    if "sig_b64" not in container_with_sig:
        return False
    sig = base64.b64decode(container_with_sig["sig_b64"])
    container = dict(container_with_sig)
    container.pop("sig_b64", None)
    data = canonical_json(container)
    try:
        pub.verify(sig, data)
        return True
    except Exception:
        return False


MAGIC = b"AUDSIG1"
HEADER_LEN = 7 + 4 + 4


def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)

    if bits.size % 8 != 0:
        pad = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    arr = np.packbits(bits)
    return arr.tobytes()


def build_frame(payload: bytes) -> bytes:
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return MAGIC + len(payload).to_bytes(4, "big") + crc.to_bytes(4, "big") + payload


def parse_frame(frame: bytes) -> Tuple[Optional[bytes], str]:
    if len(frame) < HEADER_LEN:
        return None, "Frame too small"
    if frame[:7] != MAGIC:
        return None, "Magic mismatch"
    ln = int.from_bytes(frame[7:11], "big")
    crc = int.from_bytes(frame[11:15], "big")
    payload = frame[15:15 + ln]
    if len(payload) != ln:
        return None, "Payload length mismatch"
    if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        return None, "CRC mismatch"
    return payload, "OK"


def embed_lsb_pcm16(pcm: np.ndarray, payload_bytes: bytes, repeat: int = 1) -> np.ndarray:
    frame = build_frame(payload_bytes)
    bits = bytes_to_bits(frame)
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    if bits.size > pcm.size:
        raise ValueError(f"Audio too short for payload: need {bits.size} samples, have {pcm.size}")

    out = pcm.copy()

    out[:bits.size] = (out[:bits.size] & ~1) | bits.astype(np.int16)
    return out


def extract_lsb_pcm16(pcm: np.ndarray, repeat: int = 1) -> Tuple[Optional[bytes], str]:

    header_bits_len = HEADER_LEN * 8
    if repeat > 1:
        header_bits_len_rep = header_bits_len * repeat
    else:
        header_bits_len_rep = header_bits_len

    if pcm.size < header_bits_len_rep:
        return None, "Audio too short to contain header"

    raw_bits = (pcm[:header_bits_len_rep] & 1).astype(np.uint8)

    if repeat > 1:
        raw_bits = raw_bits.reshape((-1, repeat))
        header_bits = (raw_bits.sum(axis=1) >= (repeat / 2)).astype(np.uint8)
    else:
        header_bits = raw_bits

    header_bytes = bits_to_bytes(header_bits)[:HEADER_LEN]
    if header_bytes[:7] != MAGIC:
        return None, "Signature container not found (magic mismatch)"

    ln = int.from_bytes(header_bytes[7:11], "big")
    total_frame_bytes = HEADER_LEN + ln
    total_frame_bits = total_frame_bytes * 8
    total_frame_bits_rep = total_frame_bits * repeat

    if pcm.size < total_frame_bits_rep:
        return None, "Audio too short to contain full payload"

    raw_bits2 = (pcm[:total_frame_bits_rep] & 1).astype(np.uint8)
    if repeat > 1:
        raw_bits2 = raw_bits2.reshape((-1, repeat))
        bits = (raw_bits2.sum(axis=1) >= (repeat / 2)).astype(np.uint8)
    else:
        bits = raw_bits2

    frame_bytes = bits_to_bytes(bits)[:total_frame_bytes]
    payload, status = parse_frame(frame_bytes)
    return payload, status


def plot_wave_and_spec(samples: np.ndarray, sr: int, title_prefix: str):
    st.subheader(f"{title_prefix}: Вейвформ и спектрограмма")

    fig1 = plt.figure()
    t = np.arange(samples.size) / float(sr)
    plt.plot(t, samples, linewidth=0.8)
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title(f"{title_prefix} — форма сигнала")
    st.pyplot(fig1)
    plt.close(fig1)

    fig2 = plt.figure()
    plt.specgram(samples, NFFT=1024, Fs=sr, noverlap=512)
    plt.xlabel("Время, с")
    plt.ylabel("Частота, Гц")
    plt.title(f"{title_prefix} — спектрограмма")
    st.pyplot(fig2)
    plt.close(fig2)


st.set_page_config(page_title="Audio e-Sign Embedder (Python)", layout="wide")
st.title("Встраивание электронной подписи в аудиозаписи (один файл, Python)")

st.markdown(
    """
**Что делает приложение:**
- Формирует контейнер: метаданные + аудио-фингерпринт + подпись Ed25519
- Встраивает контейнер в WAV (PCM16) через LSB
- Извлекает контейнер и проверяет подпись
- Показывает PSNR / SNR / MSE и графики (вейвформ/спектрограмма)

⚠️ Для демонстрации выбран LSB-метод (наглядный и простой). Для реальной устойчивости к MP3-сжатию обычно применяют частотные методы и коды коррекции ошибок.
"""
)

tab1, tab2, tab3 = st.tabs(["1) Встроить подпись", "2) Проверить подпись", "3) Тест устойчивости"])

with tab1:
    st.header("Встраивание подписи")

    colA, colB = st.columns(2)

    with colA:
        up = st.file_uploader("Загрузите аудиофайл (WAV предпочтительно; MP3 при наличии ffmpeg)", type=["wav", "mp3", "flac", "ogg"])
        fio = st.text_input("ФИО автора", value="")
        title = st.text_input("Название/описание аудиозаписи", value="")
        comment = st.text_area("Комментарий (опционально)", value="")
        alg_version = st.number_input("Версия алгоритма", min_value=1, max_value=99, value=1)
        repeat = st.slider("Избыточность (повторение каждого бита, повышает устойчивость)", min_value=1, max_value=9, value=3, step=2)

    with colB:
        st.subheader("Ключи Ed25519")
        if "priv_pem" not in st.session_state or "pub_pem" not in st.session_state:
            priv_pem, pub_pem = generate_keypair()
            st.session_state["priv_pem"] = priv_pem
            st.session_state["pub_pem"] = pub_pem

        st.code(st.session_state["pub_pem"].decode("utf-8"), language="text")
        with st.expander("Показать приватный ключ (не публикуйте его в реальных проектах!)"):
            st.code(st.session_state["priv_pem"].decode("utf-8"), language="text")

        if st.button("Сгенерировать новую пару ключей"):
            priv_pem, pub_pem = generate_keypair()
            st.session_state["priv_pem"] = priv_pem
            st.session_state["pub_pem"] = pub_pem
            st.rerun()

    if up is not None:
        try:
            audio = read_audio(up.getvalue(), up.name)
            st.success(f"Файл прочитан: sr={audio.sr} Гц, длительность={audio.samples.size/audio.sr:.2f} c, формат={audio.orig_format.upper()}")


            plot_wave_and_spec(audio.samples, audio.sr, "Исходный сигнал")

            if st.button("Встроить подпись в аудио"):

                fp = audio_fingerprint(audio.samples, audio.sr)

                meta = {
                    "fio": fio.strip(),
                    "title": title.strip(),
                    "comment": comment.strip(),
                }
                container = build_payload_container(meta=meta, fingerprint_bytes=fp, alg_version=int(alg_version))
                container_signed = sign_container(st.session_state["priv_pem"], container)
                payload_bytes = canonical_json(container_signed)


                pcm = float_to_int16(audio.samples)
                watermarked_pcm = embed_lsb_pcm16(pcm, payload_bytes, repeat=int(repeat))
                watermarked = int16_to_float(watermarked_pcm)


                m = mse(audio.samples, watermarked)
                s = snr_db(audio.samples, watermarked)
                p = psnr_db(audio.samples, watermarked, peak=1.0)

                st.subheader("Метрики качества после встраивания")
                st.write(f"MSE = {m:.8e}")
                st.write(f"SNR = {s:.2f} dB")
                st.write(f"PSNR = {p:.2f} dB")

                plot_wave_and_spec(watermarked, audio.sr, "Сигнал после встраивания")

                out_wav = write_wav_bytes(watermarked, audio.sr)
                st.download_button(
                    "Скачать WAV с подписью",
                    data=out_wav,
                    file_name="signed_audio.wav",
                    mime="audio/wav",
                )

                # Also offer payload preview
                with st.expander("Показать содержимое контейнера (JSON)"):
                    st.json(container_signed)

        except Exception as e:
            st.error(f"Ошибка обработки: {e}")


with tab2:
    st.header("Проверка подписи")

    up2 = st.file_uploader("Загрузите WAV с подписью (или другой, если в нём реально есть контейнер)", type=["wav", "mp3", "flac", "ogg"], key="verify_uploader")
    repeat2 = st.slider("Избыточность при извлечении (должна совпадать с тем, что использовали при встраивании)", 1, 9, 3, 2, key="rep2")

    pub_pem_input = st.text_area(
        "Открытый ключ (PEM) для проверки (можно оставить текущий из вкладки 1)",
        value=st.session_state.get("pub_pem", b"").decode("utf-8") if "pub_pem" in st.session_state else "",
        height=180,
    )

    if up2 is not None:
        try:
            audio2 = read_audio(up2.getvalue(), up2.name)
            st.info(f"Файл: sr={audio2.sr} Гц, длительность={audio2.samples.size/audio2.sr:.2f} c, формат={audio2.orig_format.upper()}")

            pcm2 = float_to_int16(audio2.samples)
            payload, status = extract_lsb_pcm16(pcm2, repeat=int(repeat2))
            st.write(f"Статус извлечения: **{status}**")

            if payload is None:
                st.warning("Контейнер не извлечён. Проверьте, что это WAV с подписью и правильный repeat.")
            else:
                container_signed = json.loads(payload.decode("utf-8"))

                st.subheader("Извлечённый контейнер")
                st.json(container_signed)

                pub_pem = pub_pem_input.encode("utf-8")
                ok = verify_container(pub_pem, container_signed)

                fp_now = audio_fingerprint(audio2.samples, audio2.sr)
                fp_now_b64 = base64.b64encode(fp_now).decode("ascii")
                fp_embedded = container_signed.get("fp_b64", "")

                st.subheader("Результаты проверки")
                st.write("Криптографическая подпись:", "✅ корректна" if ok else "❌ не прошла проверку")
                st.write("Сравнение фингерпринта:", "✅ совпадает" if fp_now_b64 == fp_embedded else "⚠️ отличается (возможны преобразования/изменения)")

        except Exception as e:
            st.error(f"Ошибка проверки: {e}")


with tab3:
    st.header("Простой тест устойчивости (демонстрация)")

    st.markdown(
        """
Здесь можно имитировать типовые преобразования:
- добавление слабого шума
- ресемплинг (если установлен scipy)
После преобразования пробуем извлечь контейнер и проверить подпись.
"""
    )

    up3 = st.file_uploader("Загрузите WAV с подписью (лучше)", type=["wav"], key="rob_uploader")
    repeat3 = st.slider("repeat (как при встраивании)", 1, 9, 3, 2, key="rep3")
    noise_level = st.slider("Добавить шум (амплитуда)", 0.0, 0.02, 0.003, 0.001)
    do_resample = st.checkbox("Ресемплинг: 44100 → 22050 → 44100 (если возможно)", value=False)

    pub_pem_input3 = st.text_area(
        "Открытый ключ (PEM)",
        value=st.session_state.get("pub_pem", b"").decode("utf-8") if "pub_pem" in st.session_state else "",
        height=160,
        key="pub3",
    )

    if up3 is not None:
        try:
            audio3 = read_audio(up3.getvalue(), up3.name)
            x = audio3.samples.copy()
            sr = audio3.sr

            if noise_level > 0:
                x = (x + np.random.normal(0, noise_level, size=x.shape).astype(np.float32)).clip(-1, 1)

            if do_resample:
                if resample_poly is None:
                    st.warning("scipy не найден — ресемплинг недоступен.")
                else:

                    sr2 = max(8000, sr // 2)
                    gcd1 = math.gcd(sr, sr2)
                    x = resample_poly(x, sr2 // gcd1, sr // gcd1).astype(np.float32)
                    gcd2 = math.gcd(sr2, sr)
                    x = resample_poly(x, sr // gcd2, sr2 // gcd2).astype(np.float32)


            pcm = float_to_int16(x)
            payload, status = extract_lsb_pcm16(pcm, repeat=int(repeat3))
            st.write(f"Статус извлечения после преобразований: **{status}**")

            if payload is None:
                st.error("Не удалось извлечь контейнер после преобразований.")
            else:
                container_signed = json.loads(payload.decode("utf-8"))
                ok = verify_container(pub_pem_input3.encode("utf-8"), container_signed)

                st.write("Подпись:", "✅ корректна" if ok else "❌ не прошла проверку")
                st.json(container_signed)

            m = mse(audio3.samples, x)
            s = snr_db(audio3.samples, x)
            p = psnr_db(audio3.samples, x, peak=1.0)

            st.subheader("Метрики относительно исходного (подписанного) файла")
            st.write(f"MSE = {m:.8e}")
            st.write(f"SNR = {s:.2f} dB")
            st.write(f"PSNR = {p:.2f} dB")

            plot_wave_and_spec(x, sr, "После преобразований")

            out_wav = write_wav_bytes(x, sr)
            st.download_button(
                "Скачать преобразованный WAV",
                data=out_wav,
                file_name="transformed.wav",
                mime="audio/wav",
            )

        except Exception as e:
            st.error(f"Ошибка: {e}")
