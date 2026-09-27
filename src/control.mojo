"""Dense kernels for state-space and transfer-function evaluation."""

from max.algorithm import parallelize
from std.sys import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime FVec = SIMD[DType.float64, W]
comptime ZERO = SIMD[DType.float64, W](0.0)
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime MAX_WORKERS = 32
comptime CHUNKS = 64
comptime TF_PARALLEL_WORK = 12_000_000
comptime SS_PARALLEL_WORK = 6_000_000


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def fp_off(addr: Int, offset: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr + offset * 8)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@always_inline
def tf_eval_range(
    num: FPtr,
    den: FPtr,
    nnum: Int,
    nden: Int,
    xr: FPtr,
    xi: FPtr,
    real: FPtr,
    imag: FPtr,
    start: Int,
    end: Int,
):
    var k = start
    while k + W <= end:
        var xrv = xr.unsafe_load[width=W](k)
        var xiv = xi.unsafe_load[width=W](k)
        var nr = FVec(num[0])
        var ni = ZERO
        for j in range(1, nnum):
            var old_nr = nr
            nr = old_nr * xrv - ni * xiv + FVec(num[j])
            ni = old_nr * xiv + ni * xrv
        var dr = FVec(den[0])
        var di = ZERO
        for j in range(1, nden):
            var old_dr = dr
            dr = old_dr * xrv - di * xiv + FVec(den[j])
            di = old_dr * xiv + di * xrv
        var scale = dr * dr + di * di
        real.unsafe_store(k, (nr * dr + ni * di) / scale)
        imag.unsafe_store(k, (ni * dr - nr * di) / scale)
        k += W

    while k < end:
        var xrv = xr.unsafe_load(k)
        var xiv = xi.unsafe_load(k)
        var nr = num[0]
        var ni = 0.0
        for j in range(1, nnum):
            var old_nr = nr
            nr = old_nr * xrv - ni * xiv + num[j]
            ni = old_nr * xiv + ni * xrv
        var dr = den[0]
        var di = 0.0
        for j in range(1, nden):
            var old_dr = dr
            dr = old_dr * xrv - di * xiv + den[j]
            di = old_dr * xiv + di * xrv
        var scale = dr * dr + di * di
        real.unsafe_store(k, (nr * dr + ni * di) / scale)
        imag.unsafe_store(k, (ni * dr - nr * di) / scale)
        k += 1


@export("mctl_chunk_count")
def mctl_chunk_count() abi("C") -> Int:
    return CHUNKS


@export("mctl_tf_eval")
def mctl_tf_eval(
    num_addr: Int,
    den_addr: Int,
    nnum: Int,
    nden: Int,
    xr_addr: Int,
    xi_addr: Int,
    real_addr: Int,
    imag_addr: Int,
    count: Int,
) abi("C"):
    var num = fp(num_addr)
    var den = fp(den_addr)
    var xr = fp(xr_addr)
    var xi = fp(xi_addr)
    var real = fp(real_addr)
    var imag = fp(imag_addr)
    if count < 1 or count * (nnum + nden) < TF_PARALLEL_WORK:
        tf_eval_range(num, den, nnum, nden, xr, xi, real, imag, 0, count)
        return

    var chunks = CHUNKS
    if chunks > count:
        chunks = count
    var workers = min(chunks, MAX_WORKERS)

    def work(ci: Int) {imm}:
        var start = count * ci // chunks
        var end = count * (ci + 1) // chunks
        tf_eval_range(num, den, nnum, nden, xr, xi, real, imag, start, end)

    parallelize(work, chunks, workers)


@always_inline
def ss_assemble(
    a: FPtr,
    b: FPtr,
    n: Int,
    m: Int,
    xrv: Float64,
    xiv: Float64,
    mat_r: FPtr,
    mat_i: FPtr,
    rhs_r: FPtr,
    rhs_i: FPtr,
):
    var total = n * n
    var t = 0
    while t + W <= total:
        mat_r.unsafe_store(t, -a.unsafe_load[width=W](t))
        mat_i.unsafe_store(t, ZERO)
        t += W
    while t < total:
        mat_r.unsafe_store(t, -a.unsafe_load(t))
        mat_i.unsafe_store(t, 0.0)
        t += 1
    for i in range(n):
        mat_r.unsafe_store(i * n + i, xrv - a.unsafe_load(i * n + i))
        mat_i.unsafe_store(i * n + i, xiv)
    var rlen = n * m
    t = 0
    while t + W <= rlen:
        rhs_r.unsafe_store(t, b.unsafe_load[width=W](t))
        rhs_i.unsafe_store(t, ZERO)
        t += W
    while t < rlen:
        rhs_r.unsafe_store(t, b.unsafe_load(t))
        rhs_i.unsafe_store(t, 0.0)
        t += 1


@always_inline
def ss_eliminate(mat_r: FPtr, mat_i: FPtr, rhs_r: FPtr, rhs_i: FPtr, n: Int, m: Int) -> Bool:
    for k in range(n):
        var krow = k * n
        var pivot = k
        var dk_r = mat_r.unsafe_load(krow + k)
        var dk_i = mat_i.unsafe_load(krow + k)
        var pivot_norm = dk_r * dk_r + dk_i * dk_i
        for i in range(k + 1, n):
            var row = i * n + k
            var cr = mat_r.unsafe_load(row)
            var ci = mat_i.unsafe_load(row)
            var candidate = cr * cr + ci * ci
            if candidate > pivot_norm:
                pivot = i
                pivot_norm = candidate
        if pivot_norm <= 1.0e-30:
            return False
        if pivot != k:
            var prow = pivot * n
            for j in range(k, n):
                var kj = krow + j
                var pj = prow + j
                var tr = mat_r.unsafe_load(kj)
                var ti = mat_i.unsafe_load(kj)
                mat_r.unsafe_store(kj, mat_r.unsafe_load(pj))
                mat_i.unsafe_store(kj, mat_i.unsafe_load(pj))
                mat_r.unsafe_store(pj, tr)
                mat_i.unsafe_store(pj, ti)
            for q in range(m):
                var kq = k * m + q
                var pq = pivot * m + q
                var tr = rhs_r.unsafe_load(kq)
                var ti = rhs_i.unsafe_load(kq)
                rhs_r.unsafe_store(kq, rhs_r.unsafe_load(pq))
                rhs_i.unsafe_store(kq, rhs_i.unsafe_load(pq))
                rhs_r.unsafe_store(pq, tr)
                rhs_i.unsafe_store(pq, ti)

        var pr = mat_r.unsafe_load(krow + k)
        var pi = mat_i.unsafe_load(krow + k)
        var pscale = pr * pr + pi * pi
        for i in range(k + 1, n):
            var irow = i * n
            var ik = irow + k
            var cr = mat_r.unsafe_load(ik)
            var ci = mat_i.unsafe_load(ik)
            var fr = (cr * pr + ci * pi) / pscale
            var fi = (ci * pr - cr * pi) / pscale
            var frv = FVec(fr)
            var fiv = FVec(fi)
            var frs = fr
            var fis = fi
            var j = k + 1
            while j + W <= n:
                var ur = mat_r.unsafe_load[width=W](krow + j)
                var ui = mat_i.unsafe_load[width=W](krow + j)
                var crv = mat_r.unsafe_load[width=W](irow + j)
                var civ = mat_i.unsafe_load[width=W](irow + j)
                mat_r.unsafe_store(irow + j, crv - (frv * ur - fiv * ui))
                mat_i.unsafe_store(irow + j, civ - (frv * ui + fiv * ur))
                j += W
            while j < n:
                var ur = mat_r.unsafe_load(krow + j)
                var ui = mat_i.unsafe_load(krow + j)
                mat_r.unsafe_store(
                    irow + j, mat_r.unsafe_load(irow + j) - (frs * ur - fis * ui)
                )
                mat_i.unsafe_store(
                    irow + j, mat_i.unsafe_load(irow + j) - (frs * ui + fis * ur)
                )
                j += 1
            var brow = i * m
            for q in range(m):
                var kq = k * m + q
                var iq = brow + q
                var rr = rhs_r.unsafe_load(kq)
                var ri = rhs_i.unsafe_load(kq)
                rhs_r.unsafe_store(iq, rhs_r.unsafe_load(iq) - (frs * rr - fis * ri))
                rhs_i.unsafe_store(iq, rhs_i.unsafe_load(iq) - (frs * ri + fis * rr))
    return True


@always_inline
def ss_backsolve(mat_r: FPtr, mat_i: FPtr, rhs_r: FPtr, rhs_i: FPtr, n: Int, m: Int):
    for i in range(n - 1, -1, -1):
        var irow = i * n
        var ur = mat_r.unsafe_load(irow + i)
        var ui = mat_i.unsafe_load(irow + i)
        var uscale = ur * ur + ui * ui
        for q in range(m):
            var iq = i * m + q
            var rr = rhs_r.unsafe_load(iq)
            var ri = rhs_i.unsafe_load(iq)
            for j in range(i + 1, n):
                var mr = mat_r.unsafe_load(irow + j)
                var mi = mat_i.unsafe_load(irow + j)
                var sr = rhs_r.unsafe_load(j * m + q)
                var si = rhs_i.unsafe_load(j * m + q)
                rr -= mr * sr - mi * si
                ri -= mr * si + mi * sr
            rhs_r.unsafe_store(iq, (rr * ur + ri * ui) / uscale)
            rhs_i.unsafe_store(iq, (ri * ur - rr * ui) / uscale)


@always_inline
def ss_project(
    c: FPtr,
    d: FPtr,
    rhs_r: FPtr,
    rhs_i: FPtr,
    n: Int,
    m: Int,
    p: Int,
    real: FPtr,
    imag: FPtr,
    f: Int,
    count: Int,
    ok: Bool,
):
    if not ok:
        for q in range(p * m):
            real.unsafe_store(q * count + f, d.unsafe_load(q))
            imag.unsafe_store(q * count + f, 0.0)
        return
    for row in range(p):
        var crow = row * n
        var drow = row * m
        for q in range(m):
            var out = (drow + q) * count + f
            var vr = d.unsafe_load(drow + q)
            var vi = 0.0
            for i in range(n):
                var cv = c.unsafe_load(crow + i)
                vr += cv * rhs_r.unsafe_load(i * m + q)
                vi += cv * rhs_i.unsafe_load(i * m + q)
            real.unsafe_store(out, vr)
            imag.unsafe_store(out, vi)


@always_inline
def ss_point(
    a: FPtr,
    b: FPtr,
    c: FPtr,
    d: FPtr,
    n: Int,
    m: Int,
    p: Int,
    xrv: Float64,
    xiv: Float64,
    real: FPtr,
    imag: FPtr,
    mat_r: FPtr,
    mat_i: FPtr,
    rhs_r: FPtr,
    rhs_i: FPtr,
    status: IPtr,
    f: Int,
    count: Int,
):
    ss_assemble(a, b, n, m, xrv, xiv, mat_r, mat_i, rhs_r, rhs_i)
    if ss_eliminate(mat_r, mat_i, rhs_r, rhs_i, n, m):
        ss_backsolve(mat_r, mat_i, rhs_r, rhs_i, n, m)
        status.unsafe_store(f, 0)
    else:
        status.unsafe_store(f, 1)
    ss_project(
        c, d, rhs_r, rhs_i, n, m, p, real, imag, f, count,
        status.unsafe_load(f) == 0,
    )


@export("mctl_ss_eval")
def mctl_ss_eval(
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    d_addr: Int,
    n: Int,
    m: Int,
    p: Int,
    xr_addr: Int,
    xi_addr: Int,
    real_addr: Int,
    imag_addr: Int,
    scratch_addr: Int,
    status_addr: Int,
    count: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var c = fp(c_addr)
    var d = fp(d_addr)
    var xr = fp(xr_addr)
    var xi = fp(xi_addr)
    var real = fp(real_addr)
    var imag = fp(imag_addr)
    var status = ip(status_addr)
    var area = n * n
    var rlen = n * m
    var stride = 2 * area + 2 * rlen
    if count < 1:
        return

    if count * area * n < SS_PARALLEL_WORK:
        var mat_r = fp(scratch_addr)
        var mat_i = fp_off(scratch_addr, area)
        var rhs_r = fp_off(scratch_addr, 2 * area)
        var rhs_i = fp_off(scratch_addr, 2 * area + rlen)
        for f in range(count):
            ss_point(
                a, b, c, d, n, m, p,
                xr.unsafe_load(f), xi.unsafe_load(f),
                real, imag, mat_r, mat_i, rhs_r, rhs_i, status, f, count,
            )
        return

    var chunks = CHUNKS
    if chunks > count:
        chunks = count
    var workers = min(chunks, MAX_WORKERS)

    def work(ci: Int) {imm}:
        var base = scratch_addr + ci * stride * 8
        var mat_r = fp(base)
        var mat_i = fp_off(base, area)
        var rhs_r = fp_off(base, 2 * area)
        var rhs_i = fp_off(base, 2 * area + rlen)
        var start = count * ci // chunks
        var end = count * (ci + 1) // chunks
        for f in range(start, end):
            ss_point(
                a, b, c, d, n, m, p,
                xr.unsafe_load(f), xi.unsafe_load(f),
                real, imag, mat_r, mat_i, rhs_r, rhs_i, status, f, count,
            )

    parallelize(work, chunks, workers)


@always_inline
def input_gain(
    b0: FPtr,
    b1: FPtr,
    u: FPtr,
    row: Int,
    q: Int,
    uk: Int,
    un: Int,
    use_b1: Bool,
) -> Float64:
    var value = b0.unsafe_load(row + q) * u.unsafe_load(uk + q)
    if use_b1:
        value += b1.unsafe_load(row + q) * u.unsafe_load(un + q)
    return value


@export("mctl_ss_simulate")
def mctl_ss_simulate(
    a_addr: Int,
    b0_addr: Int,
    b1_addr: Int,
    c_addr: Int,
    d_addr: Int,
    u_addr: Int,
    x_addr: Int,
    next_addr: Int,
    y_addr: Int,
    states_addr: Int,
    steps: Int,
    n: Int,
    m: Int,
    p: Int,
    has_b1: Int,
) abi("C"):
    var a = fp(a_addr)
    var b0 = fp(b0_addr)
    var b1 = fp(b1_addr)
    var c = fp(c_addr)
    var d = fp(d_addr)
    var u = fp(u_addr)
    var x = fp(x_addr)
    var swap = fp(next_addr)
    var y = fp(y_addr)
    var states = fp(states_addr)
    var use_b1 = has_b1 != 0

    for k in range(steps):
        for i in range(n):
            states.unsafe_store(k * n + i, x.unsafe_load(i))
        for row in range(p):
            var crow = row * n
            var value = 0.0
            for i in range(n):
                value += c.unsafe_load(crow + i) * x.unsafe_load(i)
            var drow = row * m
            var kq = k * m
            for q in range(m):
                value += d.unsafe_load(drow + q) * u.unsafe_load(kq + q)
            y.unsafe_store(k * p + row, value)
        if k + 1 < steps:
            var uk = k * m
            var un = uk + m
            var i = 0
            while i + 4 <= n:
                var r0 = i * n
                var r1 = r0 + n
                var r2 = r1 + n
                var r3 = r2 + n
                var v0 = 0.0
                var v1 = 0.0
                var v2 = 0.0
                var v3 = 0.0
                for j in range(n):
                    var xv = x.unsafe_load(j)
                    v0 += a.unsafe_load(r0 + j) * xv
                    v1 += a.unsafe_load(r1 + j) * xv
                    v2 += a.unsafe_load(r2 + j) * xv
                    v3 += a.unsafe_load(r3 + j) * xv
                var q0 = i * m
                for q in range(m):
                    v0 += input_gain(b0, b1, u, q0, q, uk, un, use_b1)
                    v1 += input_gain(b0, b1, u, q0 + m, q, uk, un, use_b1)
                    v2 += input_gain(b0, b1, u, q0 + 2 * m, q, uk, un, use_b1)
                    v3 += input_gain(b0, b1, u, q0 + 3 * m, q, uk, un, use_b1)
                swap.unsafe_store(i, v0)
                swap.unsafe_store(i + 1, v1)
                swap.unsafe_store(i + 2, v2)
                swap.unsafe_store(i + 3, v3)
                i += 4
            while i < n:
                var arow = i * n
                var value = 0.0
                for j in range(n):
                    value += a.unsafe_load(arow + j) * x.unsafe_load(j)
                var brow = i * m
                for q in range(m):
                    value += input_gain(b0, b1, u, brow, q, uk, un, use_b1)
                swap.unsafe_store(i, value)
                i += 1
            var tmp = x
            x = swap
            swap = tmp
