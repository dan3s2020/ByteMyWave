using System;
using System.IO;
using System.Diagnostics;
using System.Runtime.CompilerServices;

public static unsafe class Q4KExactLutV10
{
    const int QK = 256;
    const int BLOCK_BYTES = 144;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float HalfToFloat(ushort h)
    {
        uint sign = (uint)(h & 0x8000) << 16;
        uint exp  = (uint)(h >> 10) & 0x1F;
        uint mant = (uint)h & 0x3FF;
        uint bits;

        if (exp == 0)
        {
            if (mant == 0)
            {
                bits = sign;
            }
            else
            {
                int e = -14;
                while ((mant & 0x400) == 0)
                {
                    mant <<= 1;
                    e--;
                }
                mant &= 0x3FF;
                uint exp32 = (uint)(e + 127);
                bits = sign | (exp32 << 23) | (mant << 13);
            }
        }
        else if (exp == 31)
        {
            bits = sign | 0x7F800000u | (mant << 13);
        }
        else
        {
            uint exp32 = exp + (127u - 15u);
            bits = sign | (exp32 << 23) | (mant << 13);
        }

        return *(float*)&bits;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void GetScaleMin(int j, byte* q, out int d, out int m)
    {
        if (j < 4)
        {
            d = q[j] & 63;
            m = q[j + 4] & 63;
        }
        else
        {
            d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }

    static float[] LoadFloats(string path)
    {
        byte[] b = File.ReadAllBytes(path);
        if ((b.Length & 3) != 0)
            throw new Exception("Invalid float file");

        float[] f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }

    static void BuildSums(float* x, int cols, float* sums)
    {
        int groups = cols / 32;
        for (int g = 0; g < groups; g++)
        {
            float s = 0.0f;
            int baseX = g * 32;
            for (int i = 0; i < 32; i++)
                s += x[baseX + i];
            sums[g] = s;
        }
    }

    static void Direct(byte* q4, float* x, float* sum32, float* output, int rows, int cols)
    {
        int blocksPerRow = cols / QK;
        int rowBytes = blocksPerRow * BLOCK_BYTES;

        for (int row = 0; row < rows; row++)
        {
            float acc = 0.0f;
            byte* rowQ = q4 + row * rowBytes;

            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* block = rowQ + b * BLOCK_BYTES;
                ushort hd = *(ushort*)(block + 0);
                ushort hm = *(ushort*)(block + 2);
                float d = HalfToFloat(hd);
                float dmin = HalfToFloat(hm);
                byte* scales = block + 4;
                byte* qs = block + 16;

                for (int chunk = 0; chunk < 4; chunk++)
                {
                    int is0 = chunk * 2;
                    int sc0, min0;
                    int sc1, min1;

                    GetScaleMin(is0, scales, out sc0, out min0);
                    GetScaleMin(is0 + 1, scales, out sc1, out min1);

                    float lo = 0.0f;
                    float hi = 0.0f;
                    byte* q = qs + chunk * 32;
                    int xbase = b * 256 + chunk * 64;

                    for (int l = 0; l < 32; l++)
                    {
                        byte qb = q[l];
                        lo += (qb & 15) * x[xbase + l];
                        hi += (qb >> 4) * x[xbase + 32 + l];
                    }

                    int sumBase = b * 8 + chunk * 2;
                    acc += d * sc0 * lo - dmin * min0 * sum32[sumBase];
                    acc += d * sc1 * hi - dmin * min1 * sum32[sumBase + 1];
                }
            }

            output[row] = acc;
        }
    }

    static void BuildNibbleLut(float* x, int cols, float* lut)
    {
        for (int pos = 0; pos < cols; pos++)
        {
            float xv = x[pos];
            int baseLut = pos * 16;
            for (int q = 0; q < 16; q++)
                lut[baseLut + q] = q * xv;
        }
    }

    static void NibbleLut(byte* q4, float* sum32, float* lut, float* output, int rows, int cols)
    {
        int blocksPerRow = cols / QK;
        int rowBytes = blocksPerRow * BLOCK_BYTES;

        for (int row = 0; row < rows; row++)
        {
            float acc = 0.0f;
            byte* rowQ = q4 + row * rowBytes;

            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* block = rowQ + b * BLOCK_BYTES;
                float d = HalfToFloat(*(ushort*)(block + 0));
                float dmin = HalfToFloat(*(ushort*)(block + 2));
                byte* scales = block + 4;
                byte* qs = block + 16;

                for (int chunk = 0; chunk < 4; chunk++)
                {
                    int sc0, min0;
                    int sc1, min1;
                    GetScaleMin(chunk * 2, scales, out sc0, out min0);
                    GetScaleMin(chunk * 2 + 1, scales, out sc1, out min1);

                    byte* q = qs + chunk * 32;
                    int xbase = b * 256 + chunk * 64;
                    float lo = 0.0f;
                    float hi = 0.0f;

                    for (int l = 0; l < 32; l++)
                    {
                        byte qb = q[l];
                        lo += lut[(xbase + l) * 16 + (qb & 15)];
                        hi += lut[(xbase + 32 + l) * 16 + (qb >> 4)];
                    }

                    int sumBase = b * 8 + chunk * 2;
                    acc += d * sc0 * lo - dmin * min0 * sum32[sumBase];
                    acc += d * sc1 * hi - dmin * min1 * sum32[sumBase + 1];
                }
            }

            output[row] = acc;
        }
    }

    static void BuildByteLut(float* x, int cols, float* lo, float* hi)
    {
        int groups64 = cols / 64;
        for (int g = 0; g < groups64; g++)
        {
            int xbase = g * 64;
            for (int l = 0; l < 32; l++)
            {
                float xlo = x[xbase + l];
                float xhi = x[xbase + 32 + l];
                int pair = g * 32 + l;
                int tableBase = pair * 256;

                for (int qb = 0; qb < 256; qb++)
                {
                    lo[tableBase + qb] = (qb & 15) * xlo;
                    hi[tableBase + qb] = (qb >> 4) * xhi;
                }
            }
        }
    }

    static void ByteLut(byte* q4, float* sum32, float* lutLo, float* lutHi, float* output, int rows, int cols)
    {
        int blocksPerRow = cols / QK;
        int rowBytes = blocksPerRow * BLOCK_BYTES;

        for (int row = 0; row < rows; row++)
        {
            float acc = 0.0f;
            byte* rowQ = q4 + row * rowBytes;

            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* block = rowQ + b * BLOCK_BYTES;
                float d = HalfToFloat(*(ushort*)(block + 0));
                float dmin = HalfToFloat(*(ushort*)(block + 2));
                byte* scales = block + 4;
                byte* qs = block + 16;

                for (int chunk = 0; chunk < 4; chunk++)
                {
                    int sc0, min0;
                    int sc1, min1;
                    GetScaleMin(chunk * 2, scales, out sc0, out min0);
                    GetScaleMin(chunk * 2 + 1, scales, out sc1, out min1);

                    byte* q = qs + chunk * 32;
                    int pairBase = (b * 4 + chunk) * 32;
                    float lo = 0.0f;
                    float hi = 0.0f;

                    for (int l = 0; l < 32; l++)
                    {
                        byte qb = q[l];
                        int idx = (pairBase + l) * 256 + qb;
                        lo += lutLo[idx];
                        hi += lutHi[idx];
                    }

                    int sumBase = b * 8 + chunk * 2;
                    acc += d * sc0 * lo - dmin * min0 * sum32[sumBase];
                    acc += d * sc1 * hi - dmin * min1 * sum32[sumBase + 1];
                }
            }

            output[row] = acc;
        }
    }

    static double MaxAbs(float[] a, float[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = Math.Abs((double)a[i] - (double)b[i]);
            if (d > m)
                m = d;
        }
        return m;
    }

    static double RelativeL2(float[] a, float[] b)
    {
        double se = 0;
        double ee = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            se += d * d;
            ee += (double)a[i] * a[i];
        }
        return Math.Sqrt(se / ee);
    }

    static double Cosine(float[] a, float[] b)
    {
        double ab = 0;
        double aa = 0;
        double bb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double x = a[i];
            double y = b[i];
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        return ab / Math.Sqrt(aa * bb);
    }

    public static void Run(string qPath, string xPath, string refPath, int rows, int cols, int iterations)
    {
        if (cols % 256 != 0 || cols % 64 != 0)
            throw new Exception("Invalid Q4_K geometry");

        byte[] q4 = File.ReadAllBytes(qPath);
        float[] x = LoadFloats(xPath);
        float[] reference = LoadFloats(refPath);
        int expectedBytes = rows * (cols / 256) * BLOCK_BYTES;

        if (q4.Length != expectedBytes)
            throw new Exception("Q4_K size mismatch: " + q4.Length + " != " + expectedBytes);
        if (x.Length != cols)
            throw new Exception("x mismatch");
        if (reference.Length != rows)
            throw new Exception("ref mismatch");

        float[] direct = new float[rows];
        float[] nibble = new float[rows];
        float[] bytelut = new float[rows];
        float[] sums = new float[cols / 32];
        float[] nibbleTable = new float[cols * 16];
        int pairPositions = cols / 2;
        float[] byteLo = new float[pairPositions * 256];
        float[] byteHi = new float[pairPositions * 256];

        Console.WriteLine();
        Console.WriteLine("============================================================");
        Console.WriteLine(" Q4_K EXACT LUT V10 - NO ADDITIONAL WEIGHT LOSS");
        Console.WriteLine("============================================================");
        Console.WriteLine();
        Console.WriteLine("Rows                     : {0:N0}", rows);
        Console.WriteLine("Columns                  : {0:N0}", cols);
        Console.WriteLine("Logical weights/GEMV     : {0:N0}", (long)rows * cols);
        Console.WriteLine("Original Q4_K bytes      : {0:N0}", q4.Length);
        Console.WriteLine("Original Q4_K MiB        : {0:N3}", q4.Length / 1048576.0);
        Console.WriteLine("Nibble LUT               : {0:N3} MiB", nibbleTable.Length * 4.0 / 1048576.0);
        Console.WriteLine("Byte-pair LUT            : {0:N3} MiB", (byteLo.Length + byteHi.Length) * 4.0 / 1048576.0);
        Console.WriteLine();

        fixed (byte* pq = q4)
        fixed (float* px = x)
        fixed (float* ps = sums)
        fixed (float* pd = direct)
        fixed (float* pn = nibble)
        fixed (float* pb = bytelut)
        fixed (float* pnt = nibbleTable)
        fixed (float* pbl = byteLo)
        fixed (float* pbh = byteHi)
        {
            BuildSums(px, cols, ps);
            Direct(pq, px, ps, pd, rows, cols);
            BuildNibbleLut(px, cols, pnt);
            NibbleLut(pq, ps, pnt, pn, rows, cols);
            BuildByteLut(px, cols, pbl, pbh);
            ByteLut(pq, ps, pbl, pbh, pb, rows, cols);
        }

        Console.WriteLine("DIRECT vs Python reference");
        Console.WriteLine("  max abs diff           : {0:G9}", MaxAbs(reference, direct));
        Console.WriteLine("  relative L2            : {0:G9}", RelativeL2(reference, direct));
        Console.WriteLine("  cosine                 : {0:G12}", Cosine(reference, direct));
        Console.WriteLine();
        Console.WriteLine("NIBBLE LUT vs DIRECT");
        Console.WriteLine("  max abs diff           : {0:G9}", MaxAbs(direct, nibble));
        Console.WriteLine("  relative L2            : {0:G9}", RelativeL2(direct, nibble));
        Console.WriteLine("  cosine                 : {0:G12}", Cosine(direct, nibble));
        Console.WriteLine();
        Console.WriteLine("BYTE LUT vs DIRECT");
        Console.WriteLine("  max abs diff           : {0:G9}", MaxAbs(direct, bytelut));
        Console.WriteLine("  relative L2            : {0:G9}", RelativeL2(direct, bytelut));
        Console.WriteLine("  cosine                 : {0:G12}", Cosine(direct, bytelut));

        for (int warm = 0; warm < 2; warm++)
        {
            fixed (byte* pq = q4)
            fixed (float* px = x)
            fixed (float* ps = sums)
            fixed (float* pd = direct)
            fixed (float* pn = nibble)
            fixed (float* pb = bytelut)
            fixed (float* pnt = nibbleTable)
            fixed (float* pbl = byteLo)
            fixed (float* pbh = byteHi)
            {
                BuildSums(px, cols, ps);
                Direct(pq, px, ps, pd, rows, cols);
                BuildNibbleLut(px, cols, pnt);
                NibbleLut(pq, ps, pnt, pn, rows, cols);
                BuildByteLut(px, cols, pbl, pbh);
                ByteLut(pq, ps, pbl, pbh, pb, rows, cols);
            }
        }

        Stopwatch sw = Stopwatch.StartNew();
        for (int iter = 0; iter < iterations; iter++)
        {
            fixed (byte* pq = q4)
            fixed (float* px = x)
            fixed (float* ps = sums)
            fixed (float* pd = direct)
            {
                BuildSums(px, cols, ps);
                Direct(pq, px, ps, pd, rows, cols);
            }
        }
        sw.Stop();
        double directMs = sw.Elapsed.TotalMilliseconds / iterations;

        sw.Restart();
        for (int iter = 0; iter < iterations; iter++)
        {
            fixed (byte* pq = q4)
            fixed (float* px = x)
            fixed (float* ps = sums)
            fixed (float* pn = nibble)
            fixed (float* pnt = nibbleTable)
            {
                BuildSums(px, cols, ps);
                BuildNibbleLut(px, cols, pnt);
                NibbleLut(pq, ps, pnt, pn, rows, cols);
            }
        }
        sw.Stop();
        double nibbleMs = sw.Elapsed.TotalMilliseconds / iterations;

        sw.Restart();
        for (int iter = 0; iter < iterations; iter++)
        {
            fixed (byte* pq = q4)
            fixed (float* px = x)
            fixed (float* ps = sums)
            fixed (float* pb = bytelut)
            fixed (float* pbl = byteLo)
            fixed (float* pbh = byteHi)
            {
                BuildSums(px, cols, ps);
                BuildByteLut(px, cols, pbl, pbh);
                ByteLut(pq, ps, pbl, pbh, pb, rows, cols);
            }
        }
        sw.Stop();
        double byteMs = sw.Elapsed.TotalMilliseconds / iterations;

        Console.WriteLine();
        Console.WriteLine("------------------------------------------------------------");
        Console.WriteLine(" CUSTOM KERNEL BENCHMARK");
        Console.WriteLine("------------------------------------------------------------");
        Console.WriteLine();
        Console.WriteLine("DIRECT original Q4_K     : {0:N3} ms/GEMV", directMs);
        Console.WriteLine("NIBBLE LUT exact         : {0:N3} ms/GEMV", nibbleMs);
        Console.WriteLine("BYTE LUT exact           : {0:N3} ms/GEMV", byteMs);
        Console.WriteLine();
        Console.WriteLine("Nibble LUT / direct      : {0:N3}x", directMs / nibbleMs);
        Console.WriteLine("Byte LUT / direct        : {0:N3}x", directMs / byteMs);
        Console.WriteLine();
        Console.WriteLine("NO centroid/codebook approximation.");
        Console.WriteLine("Original Q4_K bytes are consumed directly.");
        Console.WriteLine("============================================================");
    }
}
