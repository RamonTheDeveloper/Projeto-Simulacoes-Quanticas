"""
=======================================================================
  LABORATORIO DE EMARANHAMENTO QUANTICO
  Baseado em Qiskit + AerSimulator + Matplotlib

  Conteudo:
    1. Escolha do estado de Bell (Phi+, Phi-, Psi+, Psi-)
    2. Circuito quantico e vetor de estado
    3. PROVA 1 — Rank de Schmidt (tentativa de fatoracao)
    4. PROVA 2 — Entropia de von Neumann
    5. PROVA 3 — Criterio de Peres-Horodecki (Negatividade)
    6. PROVA 4 — Concorrencia de Wootters
    7. Correlacoes e violacao das desigualdades de Bell (CHSH)
    8. Visualizacao completa em 8 paineis (sem Q-Sphere / sem RGB)
=======================================================================
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator

sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════
#  ESTADOS DE BELL
# ═══════════════════════════════════════════════════════════════
BELL_STATES = {
    '1': {
        'nome':      'Phi+ = (|00> + |11>) / sqrt(2)',
        'latex':     'Phi+',
        'descricao': (
            'Estado padrao. Correlacao perfeita:\n'
            'medir |0> em A => B colapsa para |0>;\n'
            'medir |1> em A => B colapsa para |1>.'
        ),
        'build': lambda qc: [qc.h(0), qc.cx(0, 1)],
        'esperado_E': +1.0,
    },
    '2': {
        'nome':      'Phi- = (|00> - |11>) / sqrt(2)',
        'latex':     'Phi-',
        'descricao': (
            'Igual ao Phi+, mas com fase relativa pi\n'
            'entre |00> e |11>. Correlacao +1 em Z,\n'
            'anticorrelacao em X e Y.'
        ),
        'build': lambda qc: [qc.h(0), qc.cx(0, 1), qc.z(0)],
        'esperado_E': +1.0,
    },
    '3': {
        'nome':      'Psi+ = (|01> + |10>) / sqrt(2)',
        'latex':     'Psi+',
        'descricao': (
            'Anticorrelacionado: medir |0> em A\n'
            'implica |1> em B e vice-versa.\n'
            'Usado em protocolos de teleportacao.'
        ),
        'build': lambda qc: [qc.x(1), qc.h(0), qc.cx(0, 1)],
        'esperado_E': -1.0,
    },
    '4': {
        'nome':      'Psi- = (|01> - |10>) / sqrt(2)',
        'latex':     'Psi-',
        'descricao': (
            'Singleto de spin. Unico estado de Bell\n'
            'com simetria antisimetrica. Viola as\n'
            'desigualdades de Bell mais fortemente.'
        ),
        'build': lambda qc: [qc.x(1), qc.h(0), qc.cx(0, 1), qc.z(0)],
        'esperado_E': -1.0,
    },
}

# ═══════════════════════════════════════════════════════════════
#  SIMULACAO
# ═══════════════════════════════════════════════════════════════

def build_circuit(key):
    qc = QuantumCircuit(2)
    BELL_STATES[key]['build'](qc)
    return qc


def get_statevector(qc):
    qc2 = qc.copy()
    qc2.save_statevector()
    result = AerSimulator().run(qc2).result()
    return np.array(result.get_statevector().data)


def get_counts(qc, shots=8192):
    qc2 = qc.copy()
    cr  = ClassicalRegister(2)
    qc2.add_register(cr)
    qc2.measure([0, 1], [0, 1])
    result = AerSimulator().run(qc2, shots=shots).result()
    return result.get_counts(), shots


def get_counts_basis(qc, basis='zz', shots=8192):
    """Mede em bases diferentes para calcular correlacoes CHSH."""
    qc2 = qc.copy()
    if basis == 'xz':
        qc2.h(0)
    elif basis == 'zx':
        qc2.h(1)
    elif basis == 'xx':
        qc2.h(0); qc2.h(1)
    cr = ClassicalRegister(2)
    qc2.add_register(cr)
    qc2.measure([0, 1], [0, 1])
    result = AerSimulator().run(qc2, shots=shots).result()
    return result.get_counts()


# ═══════════════════════════════════════════════════════════════
#  PROVAS MATEMATICAS
# ═══════════════════════════════════════════════════════════════

def densidade_reduzida(sv):
    """Calcula rho_AB, rho_A e rho_B por traco parcial."""
    M      = sv.reshape(2, 2)
    rho_AB = np.outer(sv, sv.conj())

    rho_A = np.zeros((2, 2), dtype=complex)
    rho_B = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                rho_A[i, j] += M[i, k] * M[j, k].conj()
                rho_B[i, j] += M[k, i] * M[k, j].conj()
    return rho_AB, rho_A, rho_B


def prova_schmidt(sv):
    """
    PROVA 1: Decomposicao de Schmidt.
    |psi> = sum_i s_i |u_i>|v_i>
    Se rank > 1: estado NAO e produto => emaranhado.
    """
    M      = sv.reshape(2, 2)
    U, s, Vh = np.linalg.svd(M)
    rank   = int(np.sum(s > 1e-10))
    return s, rank, U, Vh


def prova_entropia(rho_A, rho_B):
    """
    PROVA 2: Entropia de von Neumann.
    S(rho_A) = -Tr(rho_A log2 rho_A)
    Estado puro => S=0. Emaranhado => S>0 (max=1 para 1 qubit).
    """
    def vn(rho):
        eigs = np.linalg.eigvalsh(rho)
        eigs = eigs[eigs > 1e-12]
        return float(-np.sum(eigs * np.log2(eigs)))
    return vn(rho_A), vn(rho_B)


def prova_negatividade(rho_AB):
    """
    PROVA 3: Criterio de Peres-Horodecki.
    Transposta parcial de rho_AB em B.
    Autovalores negativos => emaranhado
    (condicao necessaria e suficiente para 2 qubits).
    Negatividade N = sum(|lambda_i|) para lambda_i < 0.
    """
    rho   = rho_AB.reshape(2, 2, 2, 2)
    rho_TB = rho.transpose(0, 3, 2, 1).reshape(4, 4)
    eigs  = np.linalg.eigvalsh(rho_TB)
    N     = float(-np.sum(eigs[eigs < 0]))
    return N, eigs, rho_TB


def prova_concorrencia(rho_AB):
    """
    PROVA 4: Concorrencia de Wootters (1998).
    C = max(0, l1 - l2 - l3 - l4)
    onde l_i sao raizes decrescentes dos autovalores de R = rho * rho_til.
    C=0 separavel, C=1 maximamente emaranhado.
    """
    sy    = np.array([[0, -1j], [1j, 0]])
    sysy  = np.kron(sy, sy)
    rho_til = sysy @ rho_AB.conj() @ sysy
    R     = rho_AB @ rho_til
    eigs  = np.sort(np.sqrt(np.abs(np.linalg.eigvals(R).real)))[::-1]
    C     = float(max(0.0, eigs[0] - eigs[1] - eigs[2] - eigs[3]))
    return C, eigs


def pureza(rho):
    return float(np.real(np.trace(rho @ rho)))


def correlacao_E(counts, shots):
    """E(a,b) = [P(mesmo) - P(diferente)] em base dada."""
    p = {k: v / shots for k, v in counts.items()}
    p00 = p.get('00', 0); p01 = p.get('01', 0)
    p10 = p.get('10', 0); p11 = p.get('11', 0)
    return (p00 + p11) - (p01 + p10)


def chsh_parameter(qc, shots=4096):
    """
    S_CHSH = E(ZZ) - E(ZX) + E(XZ) + E(XX)
    Limite classico: |S| <= 2
    Limite quantico (Tsirelson): |S| <= 2*sqrt(2) ~ 2.828
    """
    E_zz = correlacao_E(get_counts_basis(qc, 'zz', shots), shots)
    E_zx = correlacao_E(get_counts_basis(qc, 'zx', shots), shots)
    E_xz = correlacao_E(get_counts_basis(qc, 'xz', shots), shots)
    E_xx = correlacao_E(get_counts_basis(qc, 'xx', shots), shots)
    S    = E_zz - E_zx + E_xz + E_xx
    return S, E_zz, E_zx, E_xz, E_xx


# ═══════════════════════════════════════════════════════════════
#  VISUALIZACAO
# ═══════════════════════════════════════════════════════════════

BG   = '#0d0d0d'
C1   = '#5DCAA5'   # verde  — estado |00>/|11>
C2   = '#EF9F27'   # laranja
C3   = '#7F77DD'   # roxo
C4   = '#ED93B1'   # rosa
CRED = '#E24B4A'   # vermelho — emaranhado
CGRY = '#888899'   # cinza

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    if title:
        ax.set_title(title, color='#aaaacc', fontsize=9, pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color=CGRY, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=CGRY, fontsize=8)
    ax.tick_params(colors=CGRY, labelsize=7)


def plot_laboratorio(sv, counts, shots, provas, chsh, bell_info, qc_str):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        height_ratios=[1, 1, 1],
        width_ratios=[1.1, 1.1, 1.1, 1.1, 1.4],
        hspace=0.55, wspace=0.38
    )

    labels = ['|00>', '|01>', '|10>', '|11>']

    # ── 1. Contagens de medicao ──────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, 'Contagens de medicao', ylabel='Contagens')
    vals = [counts.get(k.replace('|','').replace('>',''), 0) for k in labels]
    cols = [C1 if v == max(vals) else CGRY for v in vals]
    bars = ax.bar(labels, vals, color=cols, edgecolor='#222233', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + shots*0.01,
                f'{v/shots:.1%}', ha='center', color='white', fontsize=8)
    ax.axhline(shots/4, color='#444466', linestyle=':', linewidth=0.8)
    ax.set_ylim(0, max(vals) * 1.25)

    # ── 2. Amplitudes |c_i|^2 ────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, 'Probabilidades |c_i|^2', ylabel='|c_i|^2')
    probs = np.abs(sv)**2
    ax.bar(labels, probs, color=C3, edgecolor='#222233', linewidth=0.5)
    for i, v in enumerate(probs):
        if v > 0.01:
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', color='white', fontsize=8)
    ax.set_ylim(0, 0.75)

    # ── 3. Fases das amplitudes ───────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, 'Fases arg(c_i) [rad]', ylabel='Fase (rad)')
    fases = np.angle(sv)
    fcols = [C4 if abs(f) > 0.1 else CGRY for f in fases]
    ax.bar(labels, fases, color=fcols, edgecolor='#222233', linewidth=0.5)
    for i, v in enumerate(fases):
        if abs(v) > 0.05:
            ax.text(i, v + 0.05*np.sign(v), f'{v:.2f}',
                    ha='center', color='white', fontsize=8)
    ax.axhline(0, color='#444466', linewidth=0.5)
    ax.set_ylim(-4, 4)

    # ── 4. Coeficientes de Schmidt ────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    s_vals = provas['schmidt_vals']
    rank   = provas['schmidt_rank']
    style_ax(ax, f'Coeficientes de Schmidt\nrank={rank}  =>  '
                 f'{"EMARANHADO" if rank>1 else "SEPARAVEL"}',
             ylabel='Valor singular')
    bcols = [CRED if v > 0.1 else CGRY for v in s_vals]
    ax.bar([f's{i+1}' for i in range(len(s_vals))], s_vals,
           color=bcols, edgecolor='#222233', linewidth=0.5)
    for i, v in enumerate(s_vals):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', color='white', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.text(0.5, 0.88, f'rank = {rank}',
            transform=ax.transAxes, ha='center',
            color=CRED if rank > 1 else C1,
            fontsize=13, fontweight='bold')

    # ── 5. Resumo das 4 provas ────────────────────────────────
    ax = fig.add_subplot(gs[0, 4])
    ax.set_facecolor(BG)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    ax.set_title('Resumo das provas matematicas',
                 color='#aaaacc', fontsize=9, pad=5)

    provas_txt = [
        ('PROVA 1 — Schmidt',
         f'rank = {provas["schmidt_rank"]}',
         'rank > 1 => nao-fatoravel',
         provas['schmidt_rank'] > 1),

        ('PROVA 2 — Entropia von Neumann',
         f'S(A) = {provas["S_A"]:.4f} bits',
         'S > 0 => subsistema misto',
         provas['S_A'] > 0.01),

        ('PROVA 3 — Peres-Horodecki',
         f'N = {provas["negatividade"]:.4f}',
         'N > 0 => autovalor negativo',
         provas['negatividade'] > 0.001),

        ('PROVA 4 — Concorrencia',
         f'C = {provas["concorrencia"]:.4f}',
         'C=1 => maximamente emar.',
         provas['concorrencia'] > 0.001),
    ]

    y = 0.95
    for titulo, valor, interp, positivo in provas_txt:
        cor  = CRED if positivo else C1
        seta = 'EMARANHADO' if positivo else 'SEPARAVEL'
        ax.text(0.02, y, titulo, transform=ax.transAxes,
                color='#aaaacc', fontsize=8, fontweight='bold', va='top')
        y -= 0.055
        ax.text(0.05, y, valor, transform=ax.transAxes,
                color=cor, fontsize=11, fontweight='bold', va='top')
        ax.text(0.55, y, f'=> {seta}', transform=ax.transAxes,
                color=cor, fontsize=8, va='top')
        y -= 0.055
        ax.text(0.05, y, interp, transform=ax.transAxes,
                color=CGRY, fontsize=7, va='top', style='italic')
        y -= 0.075

    # ── 6. Matriz densidade rho_A ─────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    rho_A  = provas['rho_A']
    pur_A  = provas['pureza_A']
    im = ax.imshow(np.abs(rho_A), cmap='Blues',
                   vmin=0, vmax=0.65, aspect='equal')
    ax.set_xticks([0,1]); ax.set_xticklabels(['|0>','|1>'], color=CGRY, fontsize=8)
    ax.set_yticks([0,1]); ax.set_yticklabels(['|0>','|1>'], color=CGRY, fontsize=8)
    for i in range(2):
        for j in range(2):
            v = rho_A[i, j]
            txt = f'{v.real:.3f}' if abs(v.imag) < 1e-6 else f'{v.real:.2f}+{v.imag:.2f}i'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')
    ax.set_title(f'rho_A = Tr_B(rho_AB)\nPureza = {pur_A:.4f}',
                 color='#aaaacc', fontsize=9, pad=4)
    ax.tick_params(colors=CGRY, labelsize=7)
    fig.colorbar(im, ax=ax, shrink=0.75).ax.tick_params(colors=CGRY, labelsize=6)

    # ── 7. Matriz densidade rho_B ─────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    rho_B  = provas['rho_B']
    pur_B  = provas['pureza_B']
    im2 = ax.imshow(np.abs(rho_B), cmap='Purples',
                    vmin=0, vmax=0.65, aspect='equal')
    ax.set_xticks([0,1]); ax.set_xticklabels(['|0>','|1>'], color=CGRY, fontsize=8)
    ax.set_yticks([0,1]); ax.set_yticklabels(['|0>','|1>'], color=CGRY, fontsize=8)
    for i in range(2):
        for j in range(2):
            v = rho_B[i, j]
            txt = f'{v.real:.3f}' if abs(v.imag) < 1e-6 else f'{v.real:.2f}+{v.imag:.2f}i'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')
    ax.set_title(f'rho_B = Tr_A(rho_AB)\nPureza = {pur_B:.4f}',
                 color='#aaaacc', fontsize=9, pad=4)
    ax.tick_params(colors=CGRY, labelsize=7)
    fig.colorbar(im2, ax=ax, shrink=0.75).ax.tick_params(colors=CGRY, labelsize=6)

    # ── 8. Transposta parcial — autovalores ───────────────────
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, 'Criterio PPT\n(autovalores da transposta parcial)',
             ylabel='Autovalor')
    eigs_tp = provas['eigs_transposta']
    ecols   = [CRED if e < -1e-6 else C1 for e in eigs_tp]
    ax.bar([f'l{i+1}' for i in range(4)], eigs_tp,
           color=ecols, edgecolor='#222233', linewidth=0.5)
    for i, v in enumerate(eigs_tp):
        ax.text(i, v + 0.01*np.sign(v) if abs(v)>0.01 else 0.02,
                f'{v:.3f}', ha='center', color='white', fontsize=8)
    ax.axhline(0, color=C2, linewidth=1.0, linestyle='--', alpha=0.7)
    ax.text(0.02, 0.08, 'Vermelho < 0\n=> PPT violado\n=> EMARANHADO',
            transform=ax.transAxes, color=CRED, fontsize=7,
            bbox=dict(facecolor='#1a0a0a', alpha=0.6, edgecolor='none', pad=3))

    # ── 9. Correlacoes CHSH ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    style_ax(ax, 'Correlacoes E(a,b)\npara calculo CHSH', ylabel='E(a,b)')
    bases  = ['E(ZZ)', 'E(ZX)', 'E(XZ)', 'E(XX)']
    evals  = [chsh['E_zz'], chsh['E_zx'], chsh['E_xz'], chsh['E_xx']]
    ecols2 = [C2 if abs(v) > 0.5 else CGRY for v in evals]
    ax.bar(bases, evals, color=ecols2, edgecolor='#222233', linewidth=0.5)
    for i, v in enumerate(evals):
        ax.text(i, v + 0.04*np.sign(v) if abs(v)>0.05 else 0.04,
                f'{v:.3f}', ha='center', color='white', fontsize=8)
    ax.axhline( 0, color='#444466', linewidth=0.5)
    ax.axhline( 1, color='#333355', linewidth=0.5, linestyle=':')
    ax.axhline(-1, color='#333355', linewidth=0.5, linestyle=':')
    ax.set_ylim(-1.4, 1.4)
    ax.tick_params(axis='x', labelsize=7)

    # ── 10. Parametro CHSH — comparacao ──────────────────────
    ax = fig.add_subplot(gs[1, 4])
    ax.set_facecolor(BG)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    ax.set_title('Violacao das desigualdades de Bell (CHSH)',
                 color='#aaaacc', fontsize=9, pad=5)

    S   = chsh['S']
    lim_cl  = 2.0
    lim_qt  = 2 * np.sqrt(2)
    violou  = abs(S) > lim_cl + 0.05

    # barra comparativa
    bar_ax = ax.inset_axes([0.05, 0.55, 0.90, 0.30])
    bar_ax.set_facecolor(BG)
    bar_ax.set_xlim(0, 3.0)
    bar_ax.set_ylim(-0.5, 1.5)
    bar_ax.axis('off')
    # limite classico
    bar_ax.axvline(lim_cl, color=C2, linewidth=1.5, linestyle='--')
    bar_ax.text(lim_cl, 1.2, 'classico\n|S|<=2', ha='center',
                color=C2, fontsize=7)
    # limite de Tsirelson
    bar_ax.axvline(lim_qt, color=C1, linewidth=1.5, linestyle='--')
    bar_ax.text(lim_qt, 1.2, 'Tsirelson\n2sqrt(2)', ha='center',
                color=C1, fontsize=7)
    # valor medido
    bar_ax.barh([0], [abs(S)], height=0.6,
                color=CRED if violou else CGRY, alpha=0.8)
    bar_ax.text(abs(S)/2, 0, f'|S|={abs(S):.3f}',
                ha='center', va='center', color='white',
                fontsize=9, fontweight='bold')

    # texto de conclusao
    y2 = 0.48
    def itext(txt, cor='#ccccee', size=9, bold=False):
        nonlocal y2
        ax.text(0.05, y2, txt, transform=ax.transAxes,
                color=cor, fontsize=size,
                fontweight='bold' if bold else 'normal', va='top')
        y2 -= 0.09

    itext(f'S = E(ZZ) - E(ZX) + E(XZ) + E(XX)', CGRY, 7)
    itext(f'S = {chsh["E_zz"]:.3f} - ({chsh["E_zx"]:.3f}) + {chsh["E_xz"]:.3f} + {chsh["E_xx"]:.3f}', CGRY, 7)
    itext(f'S = {S:.4f}', C2, 12, bold=True)
    itext('')
    if violou:
        itext(f'|S| = {abs(S):.3f} > 2', CRED, 9, bold=True)
        itext('VIOLA desig. classica!', CRED, 9, bold=True)
        itext('=> Correlacoes NAO-LOCAIS', CRED, 8)
        itext('=> Confirmacao de emaranhamento', CRED, 8)
    else:
        itext(f'|S| = {abs(S):.3f} <= 2', C1, 9)
        itext('NAO viola desig. classica', CGRY, 8)

    # ── 11. Circuito quantico (texto) ─────────────────────────
    ax = fig.add_subplot(gs[2, 0:2])
    ax.set_facecolor(BG)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    ax.set_title('Circuito quantico e vetor de estado',
                 color='#aaaacc', fontsize=9, pad=4)
    ax.text(0.02, 0.92, qc_str, transform=ax.transAxes,
            color='#ccccee', fontsize=8, va='top',
            fontfamily='monospace',
            bbox=dict(facecolor='#111133', alpha=0.6,
                      edgecolor='#333355', pad=5))

    # ── 12. Interpretacao fisica ──────────────────────────────
    ax = fig.add_subplot(gs[2, 2:4])
    ax.set_facecolor(BG)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    ax.set_title('Interpretacao fisica', color='#aaaacc', fontsize=9, pad=4)

    desc = bell_info['descricao']
    S_A  = provas['S_A']
    C_val = provas['concorrencia']
    N_val = provas['negatividade']
    pur_A = provas['pureza_A']

    interp = (
        f"Estado: {bell_info['nome']}\n\n"
        f"{desc}\n\n"
        f"Interpretacao das provas:\n"
        f"  - Schmidt rank={provas['schmidt_rank']}: estado nao pode ser escrito\n"
        f"    como produto |a>|b> => emaranhamento confirmado.\n\n"
        f"  - S(rho_A) = {S_A:.4f} bits: subsistema A isolado parece\n"
        f"    'misturado', embora o estado global seja puro.\n"
        f"    Isso e impossivel em estados separaveis.\n\n"
        f"  - Negatividade N={N_val:.4f}: transposta parcial tem\n"
        f"    autovalor negativo, violando a condicao PPT.\n\n"
        f"  - Concorrencia C={C_val:.4f} (C=1 = maximamente emar.):\n"
        f"    quantifica o emaranhamento como recurso."
    )
    ax.text(0.02, 0.97, interp, transform=ax.transAxes,
            color='#ccccee', fontsize=7.5, va='top',
            fontfamily='monospace',
            bbox=dict(facecolor='#111133', alpha=0.55,
                      edgecolor='#333355', pad=5))

    # ── 13. Vetor de estado completo ──────────────────────────
    ax = fig.add_subplot(gs[2, 4])
    ax.set_facecolor(BG)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    ax.set_title('Vetor de estado e amplitudes',
                 color='#aaaacc', fontsize=9, pad=4)

    y3 = 0.94
    ax.text(0.05, y3, '|psi> = ', transform=ax.transAxes,
            color='#aaaacc', fontsize=9, va='top')
    y3 -= 0.10
    for lbl, amp in zip(labels, sv):
        r, im_v = amp.real, amp.imag
        if abs(amp) < 1e-10:
            continue
        if abs(im_v) < 1e-8:
            amp_str = f'{r:+.4f}'
        else:
            amp_str = f'({r:+.4f}{im_v:+.4f}i)'
        prob = abs(amp)**2
        ax.text(0.05, y3,
                f'{amp_str} {lbl}   [P={prob:.3f}]',
                transform=ax.transAxes,
                color=C1 if prob > 0.1 else CGRY,
                fontsize=9, va='top', fontfamily='monospace',
                fontweight='bold' if prob > 0.1 else 'normal')
        y3 -= 0.10

    y3 -= 0.05
    ax.text(0.05, y3, f'Norma: {np.linalg.norm(sv):.6f}',
            transform=ax.transAxes, color=CGRY, fontsize=8, va='top')
    y3 -= 0.09
    ax.text(0.05, y3, f'Tr(rho_A^2) = {pur_A:.4f}',
            transform=ax.transAxes, color=C3, fontsize=8, va='top')
    y3 -= 0.09
    ax.text(0.05, y3, f'Tr(rho_B^2) = {provas["pureza_B"]:.4f}',
            transform=ax.transAxes, color=C3, fontsize=8, va='top')
    y3 -= 0.09
    ax.text(0.05, y3,
            '(pureza < 1 dos subsistemas\nconfirma emaranhamento)',
            transform=ax.transAxes, color=CGRY, fontsize=7, va='top',
            style='italic')

    plt.suptitle(
        f'Laboratorio de Emaranhamento Quantico  —  {bell_info["nome"]}',
        color='#ccccee', fontsize=13, y=0.995, fontweight='bold'
    )

    fname = 'emaranhamento_laboratorio.png'
    plt.savefig(fname, dpi=140, bbox_inches='tight', facecolor=BG)
    print(f'\nFigura salva: {fname}')
    plt.show()


# ═══════════════════════════════════════════════════════════════
#  TERMINAL PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def print_sep(char='=', n=58):
    print(char * n)

def print_matrix(nome, M, indent='  '):
    print(f'{indent}Matriz {nome}:')
    for row in M:
        parts = []
        for v in row:
            r, im = v.real, v.imag
            s = f'{r:+.4f}' if abs(im) < 1e-8 else f'{r:+.4f}{im:+.4f}i'
            parts.append(f'{s:>18}')
        print(f'{indent}  [{" ".join(parts)} ]')
    print()


if __name__ == '__main__':
    print_sep()
    print('   LABORATORIO DE EMARANHAMENTO QUANTICO')
    print_sep()

    print('\nEstados de Bell disponiveis:')
    for k, s in BELL_STATES.items():
        print(f'  [{k}] {s["nome"]}')
        for linha in s['descricao'].split('\n'):
            print(f'       {linha}')
        print()

    escolha = input('Escolha o estado de Bell [1/2/3/4]: ').strip()
    if escolha not in BELL_STATES:
        print('Opcao invalida. Usando estado 1 (Phi+).')
        escolha = '1'

    bell_info = BELL_STATES[escolha]
    print_sep('-')
    print(f'Estado: {bell_info["nome"]}')
    print_sep('-')

    # simula
    qc     = build_circuit(escolha)
    qc_str = str(qc)
    print('\nCircuito quantico:')
    print(qc)

    sv             = get_statevector(qc)
    counts, shots  = get_counts(qc, shots=8192)

    print(f'\nVetor de estado: {sv}')
    print(f'Contagens (shots={shots}): {counts}')

    # ── provas ────────────────────────────────────────────────
    print_sep()
    print('  PROVAS MATEMATICAS DO EMARANHAMENTO')
    print_sep()

    rho_AB, rho_A, rho_B = densidade_reduzida(sv)

    # PROVA 1
    s_vals, rank, U, Vh = prova_schmidt(sv)
    print(f'\n[PROVA 1] Decomposicao de Schmidt')
    print(f'  |psi> = sum_i s_i |u_i>|v_i>  (SVD da matriz C_ij)')
    print(f'  Coeficientes s_i: {s_vals}')
    print(f'  Rank de Schmidt  = {rank}')
    print(f'  Conclusao: {"rank>1 => estado NAO FATORAVEL => EMARANHADO" if rank>1 else "rank=1 => SEPARAVEL"}')

    # PROVA 2
    S_A, S_B = prova_entropia(rho_A, rho_B)
    print(f'\n[PROVA 2] Entropia de von Neumann')
    print_matrix('rho_A', rho_A)
    print(f'  S(rho_A) = -Tr(rho_A log2 rho_A) = {S_A:.6f} bits')
    print(f'  S(rho_B) = {S_B:.6f} bits')
    print(f'  (max=1.0 para 1 qubit maximamente emaranhado)')
    print(f'  Conclusao: {"S>0 => subsistema A em estado MISTO => EMARANHADO" if S_A>0.01 else "S=0 => SEPARAVEL"}')

    # PROVA 3
    N_val, eigs_tp, rho_TB = prova_negatividade(rho_AB)
    print(f'\n[PROVA 3] Criterio de Peres-Horodecki (transposta parcial)')
    print(f'  Autovalores de rho^TB: {eigs_tp}')
    print(f'  Negatividade N = {N_val:.6f}')
    print(f'  Conclusao: {"N>0 => autovalor negativo => PPT violado => EMARANHADO" if N_val>0.001 else "N=0 => separavel"}')

    # PROVA 4
    C_val, eigs_R = prova_concorrencia(rho_AB)
    print(f'\n[PROVA 4] Concorrencia de Wootters')
    print(f'  C = max(0, l1-l2-l3-l4) = {C_val:.6f}')
    print(f'  (C=0 separavel, C=1 maximamente emaranhado)')
    print(f'  Conclusao: {"C>0 => EMARANHADO" if C_val>0.001 else "C=0 => SEPARAVEL"}')

    # Pureza
    pur_A = pureza(rho_A)
    pur_B = pureza(rho_B)
    print(f'\n[INFO] Pureza dos subsistemas')
    print(f'  Tr(rho_A^2) = {pur_A:.6f}  (puro=1, misto<1)')
    print(f'  Tr(rho_B^2) = {pur_B:.6f}')
    print(f'  Estado global e PURO mas subsistemas sao MISTOS => emaranhamento')

    # CHSH
    print_sep('-')
    print('  Calculando correlacoes CHSH (4 bases)...')
    S, E_zz, E_zx, E_xz, E_xx = chsh_parameter(qc, shots=4096)
    print(f'  E(ZZ)={E_zz:.4f}  E(ZX)={E_zx:.4f}  E(XZ)={E_xz:.4f}  E(XX)={E_xx:.4f}')
    print(f'  S = E(ZZ) - E(ZX) + E(XZ) + E(XX) = {S:.4f}')
    print(f'  Limite classico: |S| <= 2')
    print(f'  Limite Tsirelson: |S| <= 2*sqrt(2) = {2*np.sqrt(2):.4f}')
    if abs(S) > 2.05:
        print(f'  CONCLUSAO: |S|={abs(S):.4f} > 2 => VIOLA desigualdade classica!')
        print(f'  => Confirmacao experimental de correlacoes nao-locais')
    else:
        print(f'  CONCLUSAO: |S|={abs(S):.4f} <= 2 (nao viola nesta configuracao de bases)')

    print_sep()
    print('Gerando laboratorio visual...')

    provas = {
        'rho_A':           rho_A,
        'rho_B':           rho_B,
        'pureza_A':        pur_A,
        'pureza_B':        pur_B,
        'S_A':             S_A,
        'S_B':             S_B,
        'negatividade':    N_val,
        'eigs_transposta': eigs_tp,
        'concorrencia':    C_val,
        'schmidt_vals':    s_vals,
        'schmidt_rank':    rank,
    }
    chsh_data = {
        'S': S, 'E_zz': E_zz, 'E_zx': E_zx,
        'E_xz': E_xz, 'E_xx': E_xx,
    }

    plot_laboratorio(sv, counts, shots, provas, chsh_data, bell_info, qc_str)
    print('Concluido.')