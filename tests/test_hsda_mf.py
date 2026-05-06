import pytest
from pyhail.hsda_mf import c, f1, f2, f3, g1, g2, g3, build_mf

# ── scalar membership functions ──────────────────────────────────────────────

def test_c_returns_mf_off():
    assert c((50, 0)) == 0
    assert c((50, 0), 1.5) == pytest.approx(1.5)
    assert c((999, -3), -8.75) == pytest.approx(-8.75)

def test_f1_known_value():
    # f1((50, 0)) = -0.5 + 2.5e-3*50 + 7.5e-4*50**2 + 0
    #             = -0.5 + 0.125 + 1.875 = 1.5
    assert f1((50, 0)) == pytest.approx(1.5)

def test_f1_mf_off():
    assert f1((50, 0), 0.3) == pytest.approx(1.5 + 0.3)

def test_f1_dzdr():
    assert f1((50, 1.0)) == pytest.approx(1.5 + 1.0)

def test_f2_known_value():
    # f2((50, 0)) = 0.1*(50-50) + 0 = 0
    assert f2((50, 0)) == pytest.approx(0.0)
    # f2((60, 0)) = 0.1*(60-50) = 1.0
    assert f2((60, 0)) == pytest.approx(1.0)

def test_f2_mf_off():
    assert f2((60, 0), -0.3) == pytest.approx(1.0 - 0.3)

def test_f3_known_value():
    # f3((60, 0)) = 0.1*(60-60) = 0
    assert f3((60, 0)) == pytest.approx(0.0)
    # f3((70, 0)) = 0.1*(70-60) = 1.0
    assert f3((70, 0)) == pytest.approx(1.0)

def test_f3_mf_off():
    assert f3((70, 0), 0.3) == pytest.approx(1.3)

def test_g1_known_value():
    # g1((50, 0)) = -0.9 + 1.5e-2*50 + 5e-4*50**2 + 0
    #             = -0.9 + 0.75 + 1.25 = 1.1
    assert g1((50, 0)) == pytest.approx(1.1)

def test_g1_mf_off():
    assert g1((50, 0), 0.3) == pytest.approx(1.1 + 0.3)

def test_g2_known_value():
    # g2((50, 0)) = 0.075*(50-50) + 0 = 0
    assert g2((50, 0)) == pytest.approx(0.0)
    # g2((60, 0)) = 0.075*10 = 0.75
    assert g2((60, 0)) == pytest.approx(0.75)

def test_g3_known_value():
    # g3((60, 0)) = 0.075*(60-60) = 0
    assert g3((60, 0)) == pytest.approx(0.0)
    # g3((70, 0)) = 0.075*10 = 0.75
    assert g3((70, 0)) == pytest.approx(0.75)

# ── build_mf ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alt", [0, 1, 2, 3, 4, 5])
def test_build_mf_returns_four_values(alt):
    w, mf_h1, mf_h2, mf_h3 = build_mf(alt, zh=55.0, dzdr=0.0)
    assert w is not None
    assert mf_h1 is not None
    assert mf_h2 is not None
    assert mf_h3 is not None

@pytest.mark.parametrize("alt", [0, 1, 2, 3, 4, 5])
def test_build_mf_shapes(alt):
    w, mf_h1, mf_h2, mf_h3 = build_mf(alt, zh=55.0, dzdr=0.0)
    assert len(w) == 3
    for mf in (mf_h1, mf_h2, mf_h3):
        assert len(mf) == 3          # zh, zdr, rhv rows
        assert all(len(row) == 4 for row in mf)

@pytest.mark.parametrize("alt", [0, 1])
def test_build_mf_static_weights(alt):
    w, *_ = build_mf(alt, zh=55.0, dzdr=0.0)
    assert w == [1.0, 0.3, 0.6]

def test_build_mf_alt3_uses_g_functions():
    zh, dzdr = 55.0, 0.0
    w, mf_h1, mf_h2, mf_h3 = build_mf(3, zh=zh, dzdr=dzdr)
    const = (zh, dzdr)
    # zdr row of mf_h1 for alt=3: [g2(const,-0.3), g2(const), g1(const), g1(const,0.3)]
    assert mf_h1[1][0] == pytest.approx(g2(const, -0.3))
    assert mf_h1[1][1] == pytest.approx(g2(const))
    assert mf_h1[1][2] == pytest.approx(g1(const))
    assert mf_h1[1][3] == pytest.approx(g1(const, 0.3))

def test_build_mf_alt4_uses_f_functions():
    zh, dzdr = 55.0, 0.0
    w, mf_h1, mf_h2, mf_h3 = build_mf(4, zh=zh, dzdr=dzdr)
    const = (zh, dzdr)
    # zdr row of mf_h1 for alt=4: [f2(const,-0.3), f2(const), f1(const), f1(const,0.3)]
    assert mf_h1[1][0] == pytest.approx(f2(const, -0.3))
    assert mf_h1[1][1] == pytest.approx(f2(const))
    assert mf_h1[1][2] == pytest.approx(f1(const))
    assert mf_h1[1][3] == pytest.approx(f1(const, 0.3))

def test_build_mf_invalid_alt_returns_none():
    w, mf_h1, mf_h2, mf_h3 = build_mf(99, zh=55.0, dzdr=0.0)
    assert w is None
    assert mf_h1 is None
    assert mf_h2 is None
    assert mf_h3 is None
