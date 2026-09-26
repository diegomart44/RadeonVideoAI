"""
Automated Portable Suite Builder for RadeonVideoAI.
Uses PyInstaller to generate a 100% self-contained portable folder in dist/RadeonVideoAI/
ready to be transferred to any Windows 11 PC with an AMD Radeon GPU.
"""

import os
import sys
import shutil
import subprocess

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DIST_DIR = os.path.join(PROJECT_DIR, "dist", "RadeonVideoAI")

def run_build():
    print("=" * 70)
    print(" [*] COMPILANDO RADEONVIDEOAI SUITE PORTABLE CON PYINSTALLER")
    print("=" * 70)

    # 1. Verify spec file
    spec_path = os.path.join(PROJECT_DIR, "RadeonVideoAI.spec")
    if not os.path.isfile(spec_path):
        print(f"[ERROR] Archivo spec no encontrado en {spec_path}")
        sys.exit(1)

    # 2. Execute PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        spec_path
    ]
    print(f"[*] Ejecutando: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=PROJECT_DIR)
    if res.returncode != 0:
        print("[ERROR] Falló la compilación con PyInstaller.")
        sys.exit(res.returncode)

    # 3. Post-build asset bundling
    print("\n[*] Asegurando binarios portables y scripts en dist/RadeonVideoAI/...")
    
    # Copy ffmpeg binaries into distribution root
    for bin_name in ["ffmpeg.exe", "ffprobe.exe"]:
        src = os.path.join(PROJECT_DIR, bin_name)
        dst = os.path.join(DIST_DIR, bin_name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            print(f"    + Copiando {bin_name} -> {dst}")
            shutil.copy2(src, dst)

    # Copy launcher script
    launcher_src = os.path.join(PROJECT_DIR, "Lanzar_App.bat")
    launcher_dst = os.path.join(DIST_DIR, "Lanzar_App.bat")
    if os.path.isfile(launcher_src):
        print(f"    + Copiando Lanzar_App.bat -> {launcher_dst}")
        shutil.copy2(launcher_src, launcher_dst)

    # 4. Verify executable exists
    exe_target = os.path.join(DIST_DIR, "RadeonVideoAI.exe")
    if os.path.isfile(exe_target):
        size_mb = os.path.getsize(exe_target) / (1024 * 1024)
        print("\n" + "=" * 70)
        print(" [OK] COMPILACION COMPLETADA EXITOSAMENTE")
        print(f"     Ejecutable principal: {exe_target} ({size_mb:.2f} MB)")
        print(f"     Directorio Portable:  {DIST_DIR}")
        print("     Lanzador autónomo:    Lanzar_App.bat")
        print("=" * 70)
    else:
        print(f"[ERROR] No se encontró el ejecutable generado en {exe_target}")
        sys.exit(1)

if __name__ == "__main__":
    run_build()

