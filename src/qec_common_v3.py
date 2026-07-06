"""
qec_common_v3.py
================
Korelasyonlu gurultu modelli yuzey kodu devresi (NIHAI SURUM).

i.i.d. circuit-level gurultunun (qec_common.build_circuit) uzerine,
komsu VERI kubit ciftlerine iki-kubitli depolarizasyon kanali
DEPOLARIZE2(p_corr) enjekte eder. Bu kanal stim'in DEM cikarimina
otomatik dahil olur; boylece MWPM_aware (korelasyonu bilen) ile
MWPM_naive (bilmeyen) arasinda gercek bir fark yaratir.

SURUM GECMISI (neden "nihai"?)
------------------------------
* Ilk deneme devreyi stim iterasyonu sirasinda degistiriyordu -> bozuk.
* Ikinci deneme string manipulasyonuyla REPEAT icindeki HER TICK'ten
  sonra enjeksiyon yapiyordu -> d=3'te round basina 7 kez, asiri
  agresif; DEM'i sisiriyor ve p_corr'un etkisini boguyordu.
* Bu surum stim'in CircuitRepeatBlock API'sini kullanir ve her round'a
  TAM BIR kez enjeksiyon yapar (asagiya bakiniz). Enjeksiyon sikligi
  __main__ icindeki testle dogrulanir.
"""
from __future__ import annotations

import stim

from qec_common import build_circuit as build_circuit_iid


def _get_neighbor_data_pairs(circuit: stim.Circuit) -> list:
    """Devredeki veri kubitlerinin komsu ciftlerini doner.

    stim'in rotated surface code yerlesiminde VERI kubitleri TEK (odd)
    (x, y) koordinatlarinda -- (1,1), (3,1), ... -- OLCUM (ancilla)
    kubitleri ise cift koordinatlarda oturur. Kafeste komsu iki veri
    kubiti bir eksende 2 birim aralikla dizilir; bu yuzden komsuluk
    kriteri |dx| == 2 (ayni y) veya |dy| == 2 (ayni x)'tir.

    d=3 icin 3x3 veri kafesi -> 12 cift, d=5 icin 5x5 -> 40 cift.
    """
    qubit_coords = circuit.get_final_qubit_coordinates()
    data_qubits = [q for q, (x, y) in qubit_coords.items()
                   if x % 2 == 1 and y % 2 == 1]
    coords = {q: qubit_coords[q] for q in data_qubits}
    pairs = []
    for q1 in sorted(data_qubits):
        x1, y1 = coords[q1]
        for q2 in sorted(data_qubits):
            if q2 <= q1:
                continue
            x2, y2 = coords[q2]
            if (abs(x1 - x2) == 2 and y1 == y2) or \
               (abs(y1 - y2) == 2 and x1 == x2):
                pairs.append((q1, q2))
    return pairs


def build_circuit_correlated(distance: int, rounds: int,
                             p_iid: float, p_corr: float,
                             basis: str = "Z") -> stim.Circuit:
    """Iki kubitli depolarize korelasyonlu hata eklenmis yuzey kodu devresi.

    p_corr > 0 ise her round, her komsu veri-kubit ciftine
    DEPOLARIZE2(p_corr) uygulanir. Bu MWPM'e DEM araciligi ile
    BILDIRILEBILEN bir kanaldir; ama "kor" MWPM_naive bunu BILMEZ.

    Her round'un ILK TICK'inden sonra (Hadamard'lardan once) korelasyon
    ekler. Bu, gercekci 'ardisik shot'lar arasinda kubit drift' modelini
    temsil eder: round basina tam BIR enjeksiyon (ilk round REPEAT
    disinda oldugu icin ayrica ele alinir).
    """
    if distance % 2 == 0:
        raise ValueError(f"Mesafe tek olmali: {distance}")

    base = build_circuit_iid(distance=distance, rounds=rounds,
                             p=p_iid, basis=basis)
    if p_corr <= 0:
        return base

    pairs = _get_neighbor_data_pairs(base)
    if not pairs:
        return base

    targets = []
    for q1, q2 in pairs:
        targets.append(q1)
        targets.append(q2)
    assert len(targets) % 2 == 0, "DEPOLARIZE2 cift hedef ister"

    out = stim.Circuit()
    first_tick_seen_top = False
    for instr in base:
        if isinstance(instr, stim.CircuitRepeatBlock):
            # Round basinda (REPEAT body'nin ilk TICK'inden sonra) tek enjeksiyon
            body = instr.body_copy()
            new_body = stim.Circuit()
            injected_in_round = False
            for sub in body:
                new_body.append(sub)
                if sub.name == "TICK" and not injected_in_round:
                    new_body.append("DEPOLARIZE2", targets, p_corr)
                    injected_in_round = True
            out += new_body * instr.repeat_count
        else:
            out.append(instr)
            # Ilk round (REPEAT'in disinda) icin de bir enjeksiyon
            if instr.name == "TICK" and not first_tick_seen_top:
                out.append("DEPOLARIZE2", targets, p_corr)
                first_tick_seen_top = True
    return out


if __name__ == "__main__":
    print("=== Enjeksiyon sıklığı testi ===")
    for d in (3, 5):
        c0 = build_circuit_iid(distance=d, rounds=d, p=0.005)
        c_low = build_circuit_correlated(distance=d, rounds=d,
                                         p_iid=0.005, p_corr=0.001)
        c_med = build_circuit_correlated(distance=d, rounds=d,
                                         p_iid=0.005, p_corr=0.005)

        n0 = c0.detector_error_model(decompose_errors=True).num_errors
        n_low = c_low.detector_error_model(decompose_errors=True).num_errors
        n_med = c_med.detector_error_model(decompose_errors=True).num_errors

        # Devre stringinde kac kez DEPOLARIZE2(p_corr) gecti.
        # Beklenen (cnt_low): 2 -> biri ilk round icin REPEAT disi,
        # biri REPEAT body'sinde (body rounds-1 kez tekrarlanir).
        # DIKKAT: p_corr == p_iid oldugunda (cnt_med) sayac, stim'in
        # after_clifford_depolarization'dan gelen DEPOLARIZE2(p_iid)
        # satirlariyla KIRLENIR; bu yuzden assertion 0.001 uzerinden.
        cnt_low = str(c_low).count("DEPOLARIZE2(0.001)")
        cnt_med = str(c_med).count("DEPOLARIZE2(0.005)")

        pairs = _get_neighbor_data_pairs(c0)

        print(f"\nd={d}:")
        print(f"  i.i.d.            : DEM hata sayisi = {n0}")
        print(f"  p_corr=0.001      : DEM hata sayisi = {n_low} "
              f"(string'de {cnt_low} enjeksiyon satiri)")
        print(f"  p_corr=0.005      : DEM hata sayisi = {n_med} "
              f"(string'de {cnt_med} DEPOLARIZE2(0.005) satiri; "
              f"p_iid kanallari dahil)")
        print(f"  komsu veri cifti  : {len(pairs)}")

        assert n_low > n0, "Korelasyonlu DEM i.i.d.'den fazla hata icermeli!"
        assert cnt_low == 2, "Round basina tek enjeksiyon bekleniyordu!"
    print("\nTum testler gecti: enjeksiyon round basina tam 1 kez.")
