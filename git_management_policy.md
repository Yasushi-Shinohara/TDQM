# 新しい機能を開発
git checkout -b feature/add-login

# コードを書いてコミット
git add .
git commit -m "Add login feature"

# リモートにプッシュ
git push origin feature/add-login

# GitHubでPR作成→マージ
git checkout main
git merge feature/add-login
git push origin main

# 不要になったらブランチを削除
git branch -d feature/add-login
git push origin --delete feature/add-login
