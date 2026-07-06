"""
qec_common.py
=============
Stabilizer kuantum hata düzeltme (QEC) için ortak yardımcı fonksiyonlar.

Bu modül projenin "fizik" katmanıdır. Burada:
  1) Rotated surface code (dönmüş yüzey kodu) için bir `stim` devresi kurulur,
  2) Devreye fiziksel bir gürültü modeli (circuit-level noise) eklenir,
  3) Devreden syndrome (sendrom) ölçümleri ve gerçek logical (mantıksal) hata
     etiketleri örneklenir.

NEDEN stim?
-----------
`stim` Clifford devrelerini son derece hızlı simüle eden, QEC araştırmalarında
fiili standart olan bir kütüphanedir. Milyonlarca "shot"u saniyeler içinde
üretebildiği için, AI decoder'ı eğitmek/değerlendirmek için gereken büyük
veri kümelerini TRUBA'da pratik sürelerde hazırlayabiliriz.

NEDEN yüzey kodu (surface code)?
--------------------------------
Yüzey kodu en çok çalışılan stabilizer kodudur: yüksek hata eşiği (~%1),
sadece komşu kübit etkileşimleri ve iyi bilinen bir referans decoder'ı
(MWPM) vardır. Bu da AI decoder'ımızı adil biçimde kıyaslamamızı sağlar.

TEMEL KAVRAMLAR
---------------
- detector (dedektör): Ardışık stabilizer ölçümlerinin XOR'u. Hata yoksa 0,
  yakınında hata olduysa 1 verir. AI decoder'ın GİRDİSİ budur.
- observable (gözlemlenebilir): Deney sonunda mantıksal kübitin gerçekten
  ters dönüp dönmediği. AI decoder'ın tahmin etmeye çalıştığı ETİKET budur.
- logical error (mantıksal hata): Decoder'ın tahmini, gerçek observable
  değerinden farklıysa mantıksal hata olmuş demektir.
"""

from __future__ import annotations

import numpy as np
import stim


def build_circuit(distance: int, rounds: int, p: float,
                  basis: str = "Z") -> stim.Circuit:
    """Dönmüş yüzey kodu için gürültülü bir 'memory experiment' devresi kurar.

    Memory experiment = mantıksal kübiti hazırla, `rounds` tur boyunca
    stabilizerleri ölç, sonra oku. Decoder'ın görevi: tüm bu turlar boyunca
    biriken hataların mantıksal kübiti ters çevirip çevirmediğini bulmak.

    Parametreler
    ------------
    distance : int
        Kod mesafesi d (tek sayı: 3, 5, 7, ...). Büyük d => daha fazla
        fiziksel kübit ama daha iyi koruma. d kübitlik bir hata zinciri
        gerekir ki mantıksal hata oluşsun.
    rounds : int
        Stabilizer ölçüm turu sayısı. Genelde d'ye eşit alınır (dengeli
        space-time hacmi). Tek turda ölçüm hatası ayırt edilemez; bu yüzden
        çok turlu ölçüm gerçekçi senaryodur.
    p : float
        Fiziksel hata olasılığı ("gürültü seviyesi"). Tüm gürültü
        kanallarına aynı p uygulanır; böylece p'yi tek parametreyle
        süpürerek (sweep) eşik davranışını inceleyebiliriz.
    basis : str
        "Z" veya "X". Hangi mantıksal baz korunuyor. Varsayılan "Z".

    Dönüş
    -----
    stim.Circuit
    """
    if distance % 2 == 0:
        raise ValueError(f"Kod mesafesi tek olmalı, verilen: {distance}")
    if basis not in ("Z", "X"):
        raise ValueError(f"basis 'Z' veya 'X' olmali, verilen: {basis}")

    code_task = f"surface_code:rotated_memory_{basis.lower()}"

    # circuit-level noise: gerçekçi bir gürültü modeli. Her fiziksel işlemin
    # ardına/öncesine hata kanalları eklenir:
    #   - after_clifford_depolarization: 2-kübit kapılardan sonra depolarizasyon
    #   - after_reset_flip_probability: reset sonrası yanlış durum
    #   - before_measure_flip_probability: ölçüm öncesi okuma hatası
    #   - before_round_data_depolarization: her turda veri kübitlerinde gürültü
    circuit = stim.Circuit.generated(
        code_task,
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    return circuit


def sample_shots(circuit: stim.Circuit, num_shots: int,
                 seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Devreden (detectors, observables) çiftlerini örnekler.

    Her "shot" bağımsız bir deney denemesidir.

    Dönüş
    -----
    detectors : np.ndarray, shape (num_shots, num_detectors), dtype uint8
        AI decoder'ın girdisi (sendrom bitleri).
    observables : np.ndarray, shape (num_shots, num_observables), dtype uint8
        Gerçek mantıksal flip etiketi. memory_z/memory_x için tek sütun.
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(
        shots=num_shots, separate_observables=True
    )
    # stim bool döndürür; depolama ve PyTorch için uint8'e çeviriyoruz.
    return detectors.astype(np.uint8), observables.astype(np.uint8)


def circuit_metadata(circuit: stim.Circuit) -> dict:
    """Devre hakkında, model giriş/çıkış boyutlarını belirlemek için
    gereken meta bilgileri döndürür."""
    return {
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "num_qubits": circuit.num_qubits,
    }


def detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """MWPM (referans decoder) için gereken Detector Error Model'i çıkarır.

    `decompose_errors=True`: surface code'da bir hata olayı birden çok
    dedektörü tetikleyebilir; MWPM'in çalışması için bunlar iki-dedektörlü
    "edge"lere ayrıştırılır.
    """
    return circuit.detector_error_model(decompose_errors=True)


if __name__ == "__main__":
    # Hızlı kendi kendine test: modül doğrudan çalıştırılırsa küçük bir
    # devre kurup birkaç shot örnekler ve boyutları yazdırır.
    d, r, p = 3, 3, 0.01
    circ = build_circuit(distance=d, rounds=r, p=p)
    meta = circuit_metadata(circ)
    det, obs = sample_shots(circ, num_shots=5, seed=0)
    print(f"distance={d}, rounds={r}, p={p}")
    print(f"meta = {meta}")
    print(f"detectors shape = {det.shape}, dtype = {det.dtype}")
    print(f"observables shape = {obs.shape}")
    print(f"ornek sendrom[0] = {det[0]}")
    print(f"ornek etiket[0]  = {obs[0]}")
