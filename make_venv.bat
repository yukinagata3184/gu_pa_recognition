@rem venvを作成し、必要な外部ライブラリをインストールするbatファイル
@rem venvを作成
python -m venv venv

@rem venvをアクティベート
call venv\Scripts\activate.bat

@rem ライブラリのインストール
pip install -r requirements.txt