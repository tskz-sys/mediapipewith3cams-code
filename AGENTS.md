# AGENTS.md — Codex運用ルール（ブランチ管理 & 作業開始前の儀式）

このリポジトリでは Codex が「ブランチ作成・同期儀式・安全運用」を担当する。

---

## リポジトリ関係（固定）
- origin（チーム共有fork）: https://github.com/tskz-sys/mediapipewith3cams-code.git
- upstream（本家）:            https://github.com/ngs321/mediapipewith3cams-code.git

基本方針：
- チーム内のPR/統合は **origin** に集約
- 本家への還元は **節目で upstream にPR**

---

## 絶対ルール（Safety）
- `main` へ **直接commit / 直接push禁止**（必ずブランチ → PR）
- `origin/main` への **force push禁止**（人間が明示した場合のみ例外）
- 作業は必ず topic branch（`feat/*`, `fix/*`, `docs/*`, `chore/*`）
- コンフリクトが出たら勝手に危険な解決をしない（止まって報告し、解決案を提案）
- secrets（トークン/鍵/個人情報）をコミットしない

---

## 役割分担（推奨）
### Maintainer（例：tskz-sys）
- `origin/main` の更新（upstream取り込み、PRマージ、還元判断）
- upstreamへPRを出す（節目でまとめて）

### Developers（他メンバー）
- 自分のブランチを作って開発し、`origin/main` 宛にPRを出す
- `origin/main` を勝手にpush更新しない（ブランチpushのみ）

※ 全員がWrite権限を持っていても「main保護（PR必須）」を前提とする。

---

## 初回セットアップ（全員）
### 1) リモート確認
```bash
git remote -v
```

### 2) upstreamが無ければ追加
```bash
git remote add upstream https://github.com/ngs321/mediapipewith3cams-code.git
git fetch --all --prune
git remote -v
```

---

## 作業開始前の儀式（Ritual）
目的：ローカルと共有mainのズレを減らし、コンフリクトを先に潰す。

### A. Developers（通常メンバー用：安全・軽量）
```bash
git status --porcelain
# 何か出るなら止まる（stash/commit/discard を人間に確認）

git checkout main
git pull --rebase origin main

git fetch upstream
# 注意: Developersは origin/main を勝手にpush更新しない（Maintainerがやる）
```

### B. Maintainer（origin/mainを更新する担当）
```bash
git status --porcelain
# 何か出るなら止まる

git checkout main
git pull --rebase origin main

git fetch upstream
git merge upstream/main
# コンフリクトが出たら止まって報告（ファイル一覧、解決方針）

git push origin main
```

---

## ブランチ運用
### 命名規則
- `feat/<topic>` 新機能
- `fix/<topic>` バグ修正
- `docs/<topic>` ドキュメント
- `chore/<topic>` 雑務/小リファクタ/CI 等

`<topic>` は：
- 小文字
- ハイフン区切り
- 40文字以内  
例：`feat/3cam-calibration`

### ブランチ作成
```bash
git checkout -b feat/<topic>
```

---

## コミット運用
- 1コミットは意味のある単位（小さめ推奨）
- メッセージ例：
  - `feat: add calibration script`
  - `fix: handle missing camera device`
  - `docs: update setup steps`

```bash
git add -A
git commit -m "feat: <message>"
```

---

## push と PR
### ブランチを origin に push
```bash
git push -u origin HEAD
```

### PRの基本方針
- 通常：`feat/*` → **origin/main** にPR（チーム内でレビュー＆統合）
- 還元：まとまった節目で **origin/main → upstream/main** にPR（Maintainerが担当）

CodexはUI操作ができない前提なので、PR作成時は必ず以下を出力する：
- ブランチ名
- 変更概要（What/Why）
- テスト結果（How tested）
- PRタイトル案 & 本文テンプレ

PR本文テンプレ：
- What: 何を変えたか
- Why: なぜ必要か
- How tested: どう検証したか（コマンド/ログ）
- Notes/Risks: 影響範囲、注意点

---

## uv.lock 運用
目的：依存解決の再現性を保つ。

前提：ネットワークが必要。

手順：
1. 儀式（Developers手順）を実施
2. ブランチ作成（例：`chore/add-uv-lock` / `chore/update-uv-lock`）
3. `UV_CACHE_DIR=/tmp/uv_cache uv lock`
4. `git status --porcelain` で `uv.lock` の変更有無を確認
5. 変更が無い場合：PR不要。ブランチを削除
6. 変更がある場合：コミット → PR → squash でマージ

トラブル時の確認：
1. `git check-ignore -v uv.lock`（無視対象になっていないか）
2. `git ls-files uv.lock`（追跡されているか）

---

## ローカルチェック（存在するものだけ実行）
リポジトリに合わせて、あるものだけ走らせる（無ければスキップ）。
```bash
pytest -q || true
ruff check . || true
npm test || true
npm run lint || true
make test || true
make lint || true
```

---

## 儀式・ブランチ操作の最終レポート形式（Codexの出力）
必ず最後に以下を短く出す：
- Current branch:
- `origin/main` との差分（ahead/behind）:
- 変更サマリ（3行以内）:
- 次の推奨アクション（1〜3個）:
