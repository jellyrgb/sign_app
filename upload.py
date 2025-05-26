from huggingface_hub import login, create_repo, upload_folder

# 1. 로그인 (토큰은 입력창에 붙여넣기)
login()

# 2. 모델 repo 생성 (최신 API는 name=X ❌ → repo_id=X ✅)
create_repo(repo_id="PanHwa/vit-sign-model", repo_type="model", private=False)

# 3. 모델 폴더 업로드
upload_folder(
    repo_id="PanHwa/vit-sign-model",
    folder_path="vit_model",  # 경로는 본인의 vit_model 경로로
    repo_type="model"
)
