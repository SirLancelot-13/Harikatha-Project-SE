{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # https://devenv.sh/basics/
  env.DATA_DIR = "${config.devenv.root}/backend/data";
  # numpy's manylinux wheels need these shared libraries, which nix does not put on the default path
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];
  dotenv.enable = true; # API keys live only in .env (see WORKFLOW_AND_STORAGE.md s1.5)

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.ffmpeg # MP3 export
    pkgs.libsndfile # WAV I/O (soundfile)
    pkgs.gcc # building pyworld
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    directory = "${config.devenv.root}/backend";
    venv.enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };
  languages.javascript = {
    enable = true;
    npm.enable = true;
    npm.install.enable = true;
    directory = "${config.devenv.root}/frontend";
  };

  # https://devenv.sh/processes/
  processes.backend.exec = "cd backend && uvicorn app.main:app --reload --port 8000";
  processes.frontend.exec = "cd frontend && npm run dev";

  # https://devenv.sh/scripts/
  scripts.lint.exec = "cd $DEVENV_ROOT/backend && ruff check .";
  scripts.test-backend.exec = "cd $DEVENV_ROOT/backend && pytest";
  scripts.build-frontend.exec = "cd $DEVENV_ROOT/frontend && npm ci && npm run build";
  # Local open-source TTS server (separate environment, see local_tts/SETUP.txt). Not part of `devenv up`.
  # It must NOT use devenv's UV_PROJECT_ENVIRONMENT (that is the backend venv), hence the override.
  # On NixOS the NVIDIA driver lib is in /run/opengl-driver/lib; torch cannot find the GPU without it.
  # It needs Python 3.12 (parler-tts does not support 3.14), so the install may download that interpreter.
  scripts.local-tts-install.exec = "cd $DEVENV_ROOT/local_tts && env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=.venv UV_PYTHON_DOWNLOADS=automatic UV_PYTHON_PREFERENCE=managed uv sync --python 3.12";
  scripts.local-tts.exec = "cd $DEVENV_ROOT/local_tts && env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=.venv LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH UV_PYTHON_PREFERENCE=managed uv run --no-sync python tts_server.py";
  scripts.local-tts-check.exec = "cd $DEVENV_ROOT/local_tts && env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=.venv LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH UV_PYTHON_PREFERENCE=managed uv run --no-sync python check_gpu.py";

  # https://devenv.sh/basics/
  enterShell = ''
    python --version
    node --version
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    cd backend && ruff check . && pytest
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.ruff.enable = true;
}
