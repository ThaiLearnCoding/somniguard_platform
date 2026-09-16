import os
import sys
import shutil
import subprocess
from pathlib import Path

# Đảm bảo console Windows in được tiếng Việt UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
PROJECT_ROOT = Path(__file__).resolve().parent
TFLITE_TARGET = PROJECT_ROOT / "config" / "tflite" / "somni_guard_v2.tflite"
TFLITE_CONFIG_DIR = PROJECT_ROOT / "config" / "tflite"
AUTOGEN_DIR = PROJECT_ROOT / "autogen"
BUILD_DIR = PROJECT_ROOT / "cmake_gcc" / "build"
OUTPUT_S37 = BUILD_DIR / "base" / "xG26_Devkit.s37"
OUTPUT_GBL = PROJECT_ROOT / "xG26_Devkit.gbl"

# Công cụ Silicon Labs
USER_HOME = Path.home()
COMPILER_DIR = USER_HOME / ".silabs" / "slt" / "installs" / "conan" / "p" / "aimlcebe19bdd23e2" / "p" / "tool" / "compiler"
COMPILER_PY = COMPILER_DIR / "compiler.py"
COMMANDER_EXE = USER_HOME / ".silabs" / "slt" / "installs" / "archive" / "Simplicity Commander" / "commander.exe"
PART_NUMBER = "efr32mg26b510f3200im48"

def run_step(step_name, cmd, cwd=PROJECT_ROOT):
    print(f"\n=======================================================")
    print(f"[*] {step_name}")
    print(f"[*] Command: {' '.join(str(c) for c in cmd)}")
    print(f"[*] Working Dir: {cwd}")
    print(f"=======================================================")
    # Dùng shell=False để tránh lỗi escape dấu ngoặc kép trên Windows cmd.exe
    res = subprocess.run(cmd, cwd=cwd, shell=False)
    if res.returncode != 0:
        print(f"\n[!] LỖI: Bước '{step_name}' thất bại với mã lỗi {res.returncode}.")
        sys.exit(res.returncode)

def run_pipeline(new_tflite_path: str = None, output_gbl_path: str = None):
    # Nếu không truyền file tflite, mặc định sử dụng file somni_guard_v2.tflite hiện có
    if new_tflite_path:
        new_tflite = Path(new_tflite_path).resolve()
        if not new_tflite.exists():
            print(f"[!] Lỗi: Không tìm thấy file model nguồn tại: {new_tflite}")
            sys.exit(1)
    else:
        new_tflite = TFLITE_TARGET.resolve()
        print(f"[*] Chế độ mặc định: Sử dụng file model có sẵn trong project:")
        print(f"    {new_tflite}")
        if not new_tflite.exists():
            print(f"[!] Lỗi: File model mặc định không tồn tại tại: {new_tflite}")
            sys.exit(1)

    target_gbl = Path(output_gbl_path).resolve() if output_gbl_path else OUTPUT_GBL

    # Bước 1: Copy file tflite mới (nếu khác đường dẫn đích)
    print(f"\n[1/4] Kiểm tra và cập nhật file model...")
    if new_tflite != TFLITE_TARGET.resolve():
        shutil.copy2(new_tflite, TFLITE_TARGET)
        print(f"[+] Đã ghi đè: {new_tflite} -> {TFLITE_TARGET}")
    else:
        print(f"[+] Sử dụng trực tiếp model đích: {TFLITE_TARGET}")

    # Bước 2: Force generate code ML bằng compiler.py
    step2_cmd = [
        sys.executable,
        str(COMPILER_PY),
        "generate",
        str(TFLITE_CONFIG_DIR),
        str(AUTOGEN_DIR),
        PART_NUMBER
    ]
    # Lưu ý: compiler.py yêu cầu chạy với CWD là thư mục chứa chính nó
    run_step("2/4: Force Generate code Model (ML Compiler)", step2_cmd, cwd=COMPILER_DIR)

    # Bước 3: Build firmware bằng CMake + Ninja
    step3_cmd = ["cmake", "--build", str(BUILD_DIR), "--config", "base"]
    run_step("3/4: Build Firmware .s37 bằng CMake", step3_cmd, cwd=PROJECT_ROOT)

    if not OUTPUT_S37.exists():
        print(f"[!] Lỗi: Không tìm thấy file {OUTPUT_S37} sau khi build.")
        sys.exit(1)

    # Bước 4: Tạo file GBL bằng Simplicity Commander
    step4_cmd = [
        str(COMMANDER_EXE),
        "gbl",
        "create",
        str(target_gbl),
        "--app",
        str(OUTPUT_S37)
    ]
    run_step("4/4: Tạo file .gbl bằng Simplicity Commander", step4_cmd, cwd=PROJECT_ROOT)

    print(f"\n=======================================================")
    print(f"[SUCCESS] HOÀN TẤT PIPELINE!")
    print(f"--> File GBL: {target_gbl} ({target_gbl.stat().st_size:,} bytes)")
    print(f"=======================================================")

if __name__ == "__main__":
    # Mặc định không cần truyền tham số
    #   python build_gbl_pipeline.py                     -> dùng model mặc định & output mặc định
    #   python build_gbl_pipeline.py <model.tflite>      -> ghi đè model mới & output mặc định
    #   python build_gbl_pipeline.py <model> <out.gbl>   -> tùy chỉnh cả hai
    input_tflite = sys.argv[1] if len(sys.argv) > 1 else None
    output_gbl = sys.argv[2] if len(sys.argv) > 2 else None

    run_pipeline(input_tflite, output_gbl)
