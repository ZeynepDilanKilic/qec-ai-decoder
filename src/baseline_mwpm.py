"""
baseline_mwpm.py
================
Klasik referans decoder: Minimum Weight Perfect Matching (MWPM).

NEDEN bir referans decoder?
---------------------------
AI decoder'ın "iyi" olup olmadığını söyleyebilmek için bir kıyas noktası
gerekir. MWPM, yüzey kodu için onlarca yıldır kullanılan, iyi anlaşılmış,
neredeyse optimuma yakın klasik decoder'dır. AI decoder'ımız MWPM'e
yakın ya da daha iyi logical error rate veriyorsa, öğrenmenin işe
yaradığını söyleyebiliriz.

NASIL ÇALIŞIR (kısaca)?
-----------------------
stim, devreden bir "Detector Error Model" (DEM) çıkarır: hangi fiziksel
hatanın hangi dedektörleri tetiklediğini ve mantıksal kübiti etkileyip
etkilemediğini olasılıklarıyla listeler. PyMatching bu DEM'i bir grafa
çevirir; tetiklenen dedektörleri en düşük "ağırlıklı" (en olası) hata
zinciriyle eşleştirir ve bu zincirin mantıksal kübiti çevirip çevirmediğini
söyler.

ÖNEMLİ: MWPM eğitimsizdir. Sadece gürültü modelini bilir. AI decoder ise
veriden öğrenir; bu yüzden gürültü modeli MWPM'in varsaydığından farklı
(örn. korelasyonlu) olduğunda AI decoder avantaj sağlayabilir.
"""

from __future__ import annotations

import numpy as np
import pymatching

from qec_common import build_circuit, detector_error_model


def make_mwpm_decoder(distance: int, rounds: int, p: float) -> pymatching.Matching:
    """Verilen (d, rounds, p) icin bir MWPM decoder nesnesi kurar.

    MWPM, hangi p'ye gore kurulduysa o gurultu modelini varsayar; bu yuzden
    her gurultu seviyesi icin ayri bir Matching nesnesi olusturuyoruz
    (MWPM'e en adil sartlar: dogru gurultu modelini biliyor).
    """
    circuit = build_circuit(distance=distance, rounds=rounds, p=p)
    dem = detector_error_model(circuit)
    return pymatching.Matching.from_detector_error_model(dem)


def mwpm_logical_error_rate(matching: pymatching.Matching,
                            detectors: np.ndarray,
                            observables: np.ndarray) -> dict:
    """MWPM decoder'ini bir test kumesi uzerinde calistirir.

    Mantıksal hata = MWPM'in tahmin ettiği mantıksal flip, gerçek
    observable'dan farklıysa.

    Donus: {'logical_error_rate', 'accuracy', 'num_shots', 'num_errors'}
    """
    # decode_batch: tum shot'lari tek seferde cozer (hizli, C++ arka uc).
    predicted = matching.decode_batch(detectors)  # shape (num_shots, num_obs)
    predicted = predicted.astype(np.uint8)

    # Her shot icin: tahmin == gercek mi?
    errors = np.any(predicted != observables, axis=1)
    num_errors = int(errors.sum())
    num_shots = int(len(errors))
    ler = num_errors / num_shots
    return {
        "logical_error_rate": ler,
        "accuracy": 1.0 - ler,
        "num_shots": num_shots,
        "num_errors": num_errors,
    }


if __name__ == "__main__":
    # Kendi kendine test: d=3, p=0.01 icin MWPM'i kucuk bir kumede dene.
    from qec_common import sample_shots
    d, r, p = 3, 3, 0.01
    circ = build_circuit(distance=d, rounds=r, p=p)
    det, obs = sample_shots(circ, num_shots=20000, seed=7)
    m = make_mwpm_decoder(d, r, p)
    res = mwpm_logical_error_rate(m, det, obs)
    print(f"d={d}, p={p}: MWPM logical_error_rate = "
          f"{res['logical_error_rate']:.5f} "
          f"({res['num_errors']}/{res['num_shots']})")
