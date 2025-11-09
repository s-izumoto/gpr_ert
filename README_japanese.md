# GPR–ERT 逐次実験設計と評価

このリポジトリは、**電気比抵抗トモグラフィ（ERT）** の **ガウス過程回帰（GPR）** に基づく **逐次実験設計** を実装しています。  
不均一な比抵抗場の生成から、逆解析、そしてWenner配列を用いた基準との定量的比較までのすべての段階をカバーします。
> ℹ️ 用語の背景（ERT、GPR、逆解析、電極配列など）の概要説明は、本 README の最後にある **「📘 背景説明（Background）」セクション** を参照してください。

## 要約 (TL;DR)
- **目的:** ガウス過程回帰（GPR）に基づく逐次的な実験設定によって、地中電気探査（ERT: Electrical resistivity tomography）の測定を最適化します。これにより、測定値と過去の測定デザインをもとに、次に最も情報価値の高い測定位置を自動的に選択し、事前知識なしで対象ごとに最適な測定系列を構築できます。
- **前提:** 測定には32本の電極を用い、地表に等間隔で線状に配置します。
- **ワークフロー:** 合成2次元比抵抗フィールドの生成 → 測定値・測定デザインのデータセット生成 → GPRによる逐次測定シミュレーション → 基準法との比較
- **使用スクリプト:** `01_make_fields.py` ～ `10_plot_summary.py`
- **設定:** すべてのスクリプトは YAML 設定ファイルを使用します。
- **pyGIMLi:** オープンソースの地球物理シミュレーションライブラリであり、本ワークフローではERT測定を数値的に再現するために使用します。

## GPRベース逐次設計：性能概要

**Wenner配列を用いた基準と比較**して、**Mutual Information ベースの獲得関数**を用いた GPR 逐次設計 は、全体的な評価指標において一貫した性能向上を示しました。比較画像の例は `images_example/` フォルダーにあります。

- **大津のしきい値処理および形態学的フィルタリング後のIoU** は平均で **約8% 向上**  
- **RMSE** は **6% 改善**  
- **Pearson r** および **Spearman ρ** はそれぞれ **4% 改善**  
- **Jensen–Shannon divergence (JSD)** は **2% 改善**  
- 特に **領域の下半分（深部）** において改善が顕著で、**Pearson r** は **18% 向上**、**Spearman ρ** は **16% 向上**、**RMSE** は **9% 低下** しました。  
- **表層半分** においてもすべての指標で数パーセントの改善が見られ、**悪化した指標はありません** でした。


------------------------------------------------------------------------

## 概念的な流れ

1.  **合成フィールド生成:** 現実的な地質的不均質性を再現する比抵抗フィールドを作成。
2.  **次元削減:** PCAにより比抵抗マップを低次元潜在空間へ写像。
3.  **サンプル選択:** シミュレーション対象となる代表ケースを選択。
4.  **ウォームアップ順計算:** 標準配列構成を用いて初期ERT応答を収集。
5.  **候補拡張:** 多数の配列構成に対する応答をシミュレーション。
6.  **逐次GPR:** ウォームアップ後のGPRモデルに獲得関数を適用し、最適な次測定点を逐次選択。
7.  **逆解析:** 逐次取得データに基づいて地下比抵抗を再構築。
8.  **基準逆解析:** Wenner配列を用いた基準で同様の逆解析を実施。
9.  **評価:** 再構築結果と真値を統計・空間指標で比較。
10. **可視化:** 評価結果をグラフ化し、最終的な比較図を生成。

------------------------------------------------------------------------

## 典型的な出力

-   合成比抵抗マップ
-   PCA成分および潜在表現
-   合成測定データセット
-   逐次GPRログおよびカーネル診断
-   逆解析結果（GPR vs Wenner）
-   評価サマリーおよび可視化結果


## 主な特徴

- **ウォームアップ → アクティブGPR** の2段階ワークフロー
  -   **GPRのアクティブフェーズ開始前**に、安定化のための**ウォームアップ期間**を設けます。
  -   この期間では、**固定された測定構成（通常はWenner-α配列）**を用いてシミュレーションを実施します。
  -   得られたデータが**ウォームアップデータセット**となり、GPRモデルの初期訓練データになります。
  -   その後、**逐次GPRフェーズ**が始まり、GPRモデルが候補測定点の不確実性を推定し、**獲得関数**（例：相互情報量 Mutual Information や 期待改善 Expected Improvement）を適用して次にサンプリングすべき構成を選びます。
  -   新しい測定データを訓練集合に追加し、GPRを再学習して精度を向上させていきます。
- **特徴量化／メトリック変換:**
  - **ABMN → `[dAB, dMN, mAB, mMN]`**:
    ABMN はそれぞれ **電流を印加する電極ペア（A–B）** および **電圧を測定する電極ペア（M–N）** のインデックス番号を表す。これらの番号をユニットボックスへ正規化し、双極子距離（`mAB = |B−A|`, `mMN = |N−M|`）と双極子中心（`dAB = (A+B)/2`, `dMN = (M+N)/2`）を抽出して「距離」と「位置」を明示的に分離。
  - **設計空間メトリック変換 (`cdf+whiten`)**:
    累積分布に基づく非線形伸縮（CDFマッピング）＋ホワイトニングでスケール差と共分散を補正し、**等方的で学習しやすい特徴空間**を構築。
  - **ウォームアップ統計で標準化**:
    変換後特徴をウォームアップ区間の平均・分散で標準化し、アクティブ段階の候補にも **同一パラメータ** を適用。
- **分離可能カーネル:**
  メトリック変換後の4次元特徴空間（`[dAB, dMN, mAB, mMN]` の whiten後軸）に対して、固有値の大きい2軸と小さい2軸のそれぞれに対して、 **独立RBFカーネル** を設定（和・積いずれの結合も可）。ホワイトニングにより軸は混合されているが、**距離／位置の主要軸方向は保持** され、固有値が大きい2軸は `[dAB,dMN]`小さい2軸は `[mAB,mMN]`に対応。そのため分離カーネルが **距離成分**（`dAB`, `dMN`）と **位置成分**（`mAB`, `mMN`）という異なる **物理量を分離して扱う** ことができ、GPRが「電極間距離」と「測定位置」に応じた空間的相関をそれぞれ独立に学習する。 
- **YAML駆動の設定** により完全再現可能な実験
- **フィールドごとのログ出力**（CSV/NPZ）でカーネルハイパーパラメータ、獲得統計、逆解析を記録
- **複数配列型対応:** Wenner, Schlumberger, Dipole–Dipole, Gradient
- **獲得関数:** UCB, LCB, EI, MAXVAR, Mutual Information

------------------------------------------------------------------------


---
## 🧭 プロジェクト構造

    gpr_ert/
    ├── build/                       # 再利用可能なモジュール群とコアロジック
    │   ├── __init__.py
    │   ├── make_fields.py           # 合成2次元比抵抗マップを生成
    │   ├── fit_pca.py               # PCAを学習し、比抵抗マップを低次元空間に圧縮
    │   ├── cluster_pca.py           # PCA空間でクラスタリングを行い代表サンプルを選択
    │   ├── design.py                # ERT設計空間の列挙（ABMN生成、特徴量作成）
    │   ├── measurements_warmup.py   # 初期（ウォームアップ）Wenner配列による測定の順計算
    │   ├── measurements_post_warmup.py  # 拡張配列型の順計算
    │   ├── sequential_GPR.py        # ガウス過程回帰（GPR）のコア実装
    │   ├── invert_GPR.py            # GPR選択測定データの一括逆解析
    │   ├── forward_invert_Wenner.py # Wenner配列を用いた基準の順計算＋逆解析ワークフロー
    │   └── evaluate_GPR_vs_Wenner.py# 評価指標と比較関数
    │
    ├── scripts/                     # buildモジュールを呼び出す簡易CLIスクリプト
    │   ├── 01_make_fields.py
    │   ├── 02_fit_pca.py
    │   ├── 03_cluster_pca.py
    │   ├── 04_measurements_warmup.py
    │   ├── 05_measurements_post_warmup.py
    │   ├── 06_sequential_GPR.py
    │   ├── 07_invert_GPR.py
    │   ├── 08_forward_invert_Wenner.py
    │   ├── 09_evaluate_GPR_vs_Wenner.py
    │   ├── 10_plot_summary.py
    |   └── XX_make_images.py
    │
    ├── configs/                     # 実験を再現可能にするYAML設定ファイル
    │   ├── make_fields.yml          # 合成フィールド生成のパラメータ（ドメインサイズ、乱数性など）
    │   ├── fit_pca.yml              # PCAの設定（主成分数、寄与率など）
    │   ├── cluster_pca.yml          # クラスタリングの設定（クラスタ数、アルゴリズム、サンプリング方法）
    │   ├── measurements_warmup.yml  # Wenner配列のウォームアップ順計算設定
    │   ├── measurements_post_warmup.yml # 複数配列型の順計算設定
    │   ├── sequential_GPR.yml       # 逐次GPR設計の設定（カーネル、獲得関数、ウォームアップ長など）
    │   ├── invert_GPR.yml           # 逐次GPR出力の逆解析設定
    │   ├── forward_invert_Wenner.yml# Wenner配列を用いた基準の順計算＋逆解析設定
    │   └── evaluate_GPR_vs_Wenner.yml # 評価および比較指標設定
    │
    ├── README.md                    # プロジェクト概要とワークフロー
    └── environment.yml              # 再現性のためのPython依存関係リスト

### 説明

-   **build/** --- 科学的・計算的なコア部分を含む。\
    各モジュールは個別にインポート・テスト可能であり、全てのスクリプトがこれらのAPIを利用します。\
-   **scripts/** ---
    YAML設定を読み込み、対応する`build/`モジュールを呼び出す最小限のCLIスクリプト。\
-   **configs/** ---
    実験パラメータ群を格納。YAMLファイルを変更するだけで再現可能な実験を実行できます。


## 🛠️ インストール

このリポジトリには、環境を再現するための `environment.yml` が同梱されています。

### クイックスタート（推奨）
```bash
# 0) （任意）Mambaforge または Miniconda をインストール
#    https://conda-forge.org/miniforge/

# 1) 公開用の名前で環境を作成（ユーザー固有パスを使用しない）
mamba env create -n gpr-ert -f environment.yml    # mamba がある場合
# または
conda env create -n gpr-ert -f environment.yml

# 2) 有効化
conda activate gpr-ert
を
# 3) バージョン確認
python -V
pip -V
```

------------------------------------------------------------------------

## ⚙️ パイプラインワークフロー

このセクションでは、リポジトリに実装された**再現可能なワークフロー**をまとめます。
各ステップは順に実行され、対応する`configs/`内のYAMLファイルを使用します。
ワークフローは、**合成フィールド生成 → 逆解析・評価・可視化**までを網羅します。
ワークフローは **10個のモジュール化されたスクリプト**から構成され、それぞれが前段の出力を次の入力として受け取ります。

------------------------------------------------------------------------

1.  **合成比抵抗フィールドの生成**
    多様な地質シナリオを模擬した合成2次元比抵抗フィールドを生成。

    ``` bash
    python scripts/01_make_fields.py --config configs/make_fields.yml
    ```

2.  **PCAによる潜在空間への射影**
    PCAモデルを学習し、比抵抗マップを低次元潜在空間に圧縮・再構成。

    ``` bash
    python scripts/02_fit_pca.py --config configs/fit_pca.yml
    ```

3.  **潜在表現のクラスタリング**
    PCA後の潜在ベクトルをクラスタリングし、代表サンプルを抽出。

    ``` bash
    python scripts/03_cluster_pca.py --config configs/cluster_pca.yml
    ```

4.  **ウォームアップ測定シミュレーション（Wenner配列）**
    Wenner-α配列を用いたERT測定をpyGIMLiでシミュレーションし、GPR初期化用の**ウォームアップデータセット**を構築。

    ``` bash
    python scripts/04_measurements_warmup.py --config configs/measurements_warmup.yml
    ```

5.  **拡張測定シミュレーション（複数配列）**
    Schlumberger、Dipole-Dipole、Gradientなど複数の配列型から拡張データを生成。

    ``` bash
    python scripts/05_measurements_post_warmup.py --config configs/measurements_post_warmup.yml
    ```

6.  **逐次ガウス過程回帰（GPR）**
    ウォームアップデータで初期学習し、アクティブフェーズで獲得関数を用いて次に最も情報量の高い測定構成を選択。

    ``` bash
    python scripts/06_sequential_GPR.py --config configs/sequential_GPR.yml
    ```

7.  **逐次GPR結果の逆解析**
     逐次GPRによる測定データを逆解析し、地下比抵抗分布を再構築。

    ``` bash
    python scripts/07_invert_GPR.py --config configs/invert_GPR.yml
    ```

8.  **Wenner配列による順計算＋逆解析**
    Wenner配列を用いた基準の順計算＋逆解析を行い、参照データを作成。

    ``` bash
    python scripts/08_forward_invert_Wenner.py --config configs/forward_invert_Wenner.yml
    ```

9.  **GPRとWenner配列の比較評価**
    GPRベースとWenner配列ベースの逆解析結果を、誤差・空間的類似度などで定量評価。

    ``` bash
    python scripts/09_evaluate_GPR_vs_Wenner.py --config configs/evaluate_GPR_vs_Wenner.yml
    ```

10. **サマリープロットとレポートの生成**
    評価結果を可視化し、ボックスプロットや相対改善率などをまとめて出力。

    ```bash     
    python scripts/10_plot_summary.py --config configs/plot_summary.yml`
    ```

---

## 各スクリプトの詳細説明
このセクションでは、各スクリプトの目的と主な入出力をまとめています。ファイル名やパスは YAML の設定に従います。

### **01_make_fields.py & make_fields.py — 合成比抵抗マップ生成ステップ**  
**目的:**  
複数の地質・環境シナリオ（例: 地層構造、塩水侵入、水位変動など）に基づく**2次元の合成比抵抗マップ（log₁₀ρ）データセット**を生成します。`01_make_fields.py` がYAML設定ファイルを読み込み、指定されたパラメータを `make_fields.py` に渡して、シミュレーション的に多様な地質構造を生成します。  

**処理概要:**  
1. **YAML読込 (`01_make_fields.py`)**  
   - 設定ファイルを読み込み、解像度・サンプル数・乱数シードなどを取得。  
   - 抽出したパラメータをコマンドライン引数として `make_fields.py` に転送。  

2. **フィールド生成 (`make_fields.py`)**  
   - 指定したケース（TRACER, GEOLOGY, SURFACE, SEAWATER, WATERTABLE, RESISTIVE）ごとに、滑らかな層構造と異方的・不均質な変動を持つ log₁₀(ρ) マップを生成。  
   - 各ケースで指定数（例: 200 枚）のサンプルを生成してデータセットを構築。  
   - 各マップは上層の構造的特徴を強調し、深部では平滑化された抵抗構造を再現。  

3. **保存とメタデータ出力**  
   - 全マップを `dataset.npz`（配列: X_log10, y, cases）として圧縮保存。  
   - クラスラベル対応表 (`label_map.json`) とメタ情報 (`meta.json`) を出力。  
   - （任意）プレビュー画像を `previews/` フォルダに保存。  

**入出力:**  
- **入力:**  
  - YAML設定ファイル（例: `configs/make_fields.yml`）  
- **出力:**  
  - `dataset.npz` （形状: N×NZ×NX, 値は log₁₀(ρ[Ω·m])）  
  - `label_map.json` （ケース名 → クラスID の対応表）  
  - `meta.json` （格子サイズ・クラス情報・範囲・サンプル数など）  
  - `previews/`（任意、PNGプレビュー画像）

---

### **02_fit_pca.py & fit_pca.py — 主成分分析（PCA）による次元圧縮ステップ**  
**目的:**  
前ステップで生成した比抵抗マップ（log₁₀ρ）データセットに対して、**主成分分析（PCA）**を適用し、空間的特徴を少数の潜在変数に圧縮します。`02_fit_pca.py` は設定ファイル（YAML）を読み込み、内容をコマンドライン引数に変換して `fit_pca.py` を実行します。`fit_pca.py` は実際のPCA処理を行い、学習済み基底・投影結果・再構成プレビューなどを保存します。  

**処理概要:**  
1. **YAML読込 (`02_fit_pca.py`)**  
   - PCAの設定（入力データ、出力ディレクトリ、最大主成分数、目標分散、解法など）を読み込み。  
   - それらの値をCLI引数に変換し、`fit_pca.py` を実行。  

2. **PCAフィッティング (`fit_pca.py`)**  
   - `.npz` ファイルから `X_log10`（形状: N×NZ×NX）を読み込み、必要に応じて上部（浅部）のみを `--crop-frac` で切り出し。  
   - データを `(N, D)` にフラット化し、`PCA` または `IncrementalPCA` で次元削減を実施。  
   - 単一のPCA（全データ）またはクラス別PCA（`--per-class`）を選択可能。  
   - 各主成分の累積寄与率を計算し、`--target-var` に到達する最小次元 `k*` を自動決定。  
   - 学習済みPCA基底、メタ情報、（任意で）低次元座標や再構成画像を保存。  

3. **出力の保存**  
   - 共通PCAの場合: `pca_joint.joblib`, `pca_joint_meta.json`, `Z.npz`（任意）, `previews_joint_all/`（任意）。  
   - クラス別PCAの場合: `pca_class*.joblib`, `pca_per_class_meta.json`, `Z_per_class.npz`（任意）。  

**入出力:**  
- **入力:**  
  - `.npz` 形式のデータセット（例: `dataset.npz`）  
    - 必須: `X_log10`（N×NZ×NX）  
    - 任意: `y`（クラスラベル）, `cases`（サンプル名など）  
  - YAML設定ファイル（例: `configs/fit_pca.yml`）  
- **出力:**  
  - `pca_joint.joblib` （共通PCAの平均・基底ベクトル・寄与率など）  
  - `pca_joint_meta.json` （主要パラメータと累積寄与率）  
  - `pca_class*.joblib` （クラス別PCAの場合）  
  - `pca_per_class_meta.json` （クラスごとのメタ情報）  
  - `Z.npz` / `Z_per_class.npz` （低次元潜在表現）  
  - `previews_joint_all/` （任意、真値と再構成の比較PNG）  

---

### **03_cluster_pca.py & cluster_pca.py — PCA潜在空間のクラスタリング＆代表サンプル出力ステップ**  
**目的:**  
前ステップで得た潜在ベクトル **Z（PCAの低次元表現）** をクラスタリングして、**代表的なパターンの把握・要約**を行います。`03_cluster_pca.py` は設定（YAML）を読み込み、内容をCLI引数に変換して `cluster_pca.py` を実行します。`cluster_pca.py` は Z のクラスタリング、メタ情報・ラベルの保存、**クラスタ中心の再構成画像（centroids）**や**実サンプル代表（medoids）**の再構成画像出力、**エルボー法のkスイープ**（任意）を行います。  

**処理概要:**  
1. **YAML読込・引数転送（`03_cluster_pca.py`）**  
   - キー→フラグの固定マッピングで、`--Z`, `--pca`, `--k`, `--algo`, `--n-init`, `--max-iter`, `--random-state`, `--out` などを組み立て。  
   - 機能トグル（例: `silhouette: true` → `--silhouette`, `elbow: true` → `--elbow`）を反映し、`cluster_pca.py` を起動。  

2. **クラスタリング本体（`cluster_pca.py`）**  
   - **入力読込:** `Z.npz` の `Z (N×z_dim)` と `pca_joint.joblib`（mean, components, nz, nx, evr など）をロード。  
   - **（任意）ホワイトニング:** PCAの寄与率 `explained_variance_ratio` に基づき、各軸を `sqrt(EVR)` で割ってスケーリング。  
   - **学習:** `--algo` に応じて **KMeans / MiniBatchKMeans** を選択して学習・予測（`--k`, `--n-init`, `--max-iter`, `--random-state`）。  
   - **（任意）エルボー法:** `--elbow` 有効時に `k_min..k_max` を `k_step` で走査し、**inertia–k 曲線**（CSV/PNG）と自動推定した elbow k（JSON）を保存。  
   - **保存:** 予測ラベル（`kmeans_labels.npz`）、クラスタサイズや慣性・シルエット（任意）を含む `meta.json` を出力。  
   - **（任意）再構成出力:**  
     - **centroids/**: 各クラスタ中心（Z空間の重心）を PCA の逆変換で **X 空間（画像）に再構成**してPNG保存。  
     - **medoids/**: 各クラスタ中心に最も近い実サンプル（medoid）を選び、**実データ由来の代表画像**をPNG保存。対応表は `medoids_index.npz` に保存。  

**入出力:**  
- **入力:**  
  - 潜在表現：`Z.npz`（キー: `Z`, 形状 = N×z_dim）  
  - PCA基底：`pca_joint.joblib`（少なくとも `mean`, `components`, `nz`, `nx`。あれば `explained_variance_ratio` でwhiten可能）  
  - YAML設定ファイル（例: `configs/pca/cluster_pca.yml`）  
- **出力:**  
  - 予測ラベル：`kmeans_labels.npz`（`labels` = N,）  
  - メタ情報：`meta.json`（algo, k, inertia, sizes, whiten有無, （任意）silhouette）  
  - **（任意）エルボー:** `elbow_inertia.csv`, `elbow.png`, `elbow_meta.json`  
  - **代表画像:**  
    - **centroids/**：クラスタ中心の再構成PNG  
    - **medoids/**：各クラスタの実サンプル代表の再構成PNG  
    - `medoids_index.npz`：クラスタID・元インデックス・PNG相対パスの対応表（**任意だが強く推奨**）

**注意（推奨）:**  
- `medoids_index.npz` は後段の **GPR（Gaussian Process Regression）ベースの設計**で参照する想定のため、**出力を有効化することを推奨**します（設定ファイル／フラグで medoids 出力をオンにしてください）。

---

### **04_measurements_warmup.py & measurements_warmup.py — ERT ウォームアップ順解析（Wenner-α配列）ステップ**
**目的:**  
PCAで得た潜在表現 **Z** から浅部の log₁₀ρ を再構成し、深さ方向にパディングしたうえで **pyGIMLi** による **Wenner-α 配列**の順解析を実行し、**設計特徴量（D/Dnorm/ABMN）→ ラベル y=log10(rhoa)** の教師データ束を作成します。`04_measurements_warmup.py` は設定（YAML）を読み込み、内容をCLI引数に変換して `measurements_warmup.py` を実行します。`measurements_warmup.py` が forward シミュレーション本体を担当します。

**処理概要:**  
1. **YAML読込とCLI組み立て（`04_measurements_warmup.py`）**  
   - 既知キーのみをマッピングして CLI フラグへ変換（ブールはフラグ化、配列は繰り返し展開）。  
   - `active_policy: explicit` の場合は `active_indices` と `n_active_elec` の整合をチェック。  
   - `pattern` 未指定時は **`wenner-alpha`** を既定として `measurements_warmup.py` を起動。  

2. **順解析本体（`measurements_warmup.py`）**  
   - **PCA pack**（`mean`, `components`, `nz`, `nx`）と **Z** を読込み、`Z @ components + mean` で **浅部 log₁₀ρ（nz×nx）** を再構成 → **最下行の複写**で `nz_full` までパディング。  
   - 地表に **等間隔の固定センサー線**（`n_elec`, `dx_elec`, `margin`）を生成し、**Wenner-α** を **アクティブ部分集合**上で列挙（`every_other` / `first_k` / `explicit`）。  
   - 2D メッシュを構築 → 画像（log₁₀ρ）をセルへサンプリング → **ERT forward**（`--mode 2d/25d`）で **rhoa** を計算。  
   - **設計特徴量** `D = [dAB, dMN, mAB, mMN]`（m）と、内側幅基準の **正規化特徴量 `Dnorm`** を作成。  
   - 必要に応じて相対ガウスノイズ（`--noise-rel`）を付与し、**ラベル** `y = log10(rhoa)` を生成。  
   - 全フィールドを連結して **学習用NPZバンドル**を出力。  

3. **出力の保存（NPZ, 付随ファイル）**  
   - `ert_surrogate_wenner.npz`：  
     - `Z` (M, k) … 潜在ベクトル（設計ごとに反復）  
     - `D` (M, 4) … 設計特徴量 `[dAB,dMN,mAB,mMN]` [m]  
     - `Dnorm` (M, 4) … 内側幅 `L_inner` による 0–1 正規化  
     - `ABMN` (M, 4) int32 … 固定センサー列に対する **グローバル** 0始インデックス  
     - `y` (M,) … **log10(rhoa)**  
     - `rhoa` (M,) … 見かけ比抵抗 [Ω·m]（ノイズ付き可）  
     - `k` (M,) … 幾何学係数  
     - `xs` (n_elec,) … センサーの地表 x 座標 [m]  
     - `meta` … 幾何・パターン・シード等のJSON文字列  
   - `field_source_idx.npy`：Zスライス内での元フィールド行インデックス  

**入出力:**  
- **入力:**  
  - PCA pack（`pca_joint.joblib` 等; `mean/components/nz/nx` を含む）  
  - 潜在 `Z`（.npy/.npz; 2D 形状 N×k）  
  - YAML（ランチャー用設定; 例 `pattern`, `n_active_elec`, `active_policy`, `nz_full/nx_full`, `n_elec/dx_elec/margin`, `noise_rel`, `jobs` など）  
- **出力:**  
  - `outputs/ert_surrogate_wenner.npz`（上記一式）  
  - `outputs/field_source_idx.npy`（再現性のためのフィールド索引）  

---

### **05_measurements_post_warmup.py & measurements_post_warmup.py — ERT順解析（ウォームアップ後／複数配列対応）ステップ**
**目的:**  
PCA由来の潜在表現 **Z** をもとに浅部 log₁₀ρ を再構成し、深さ方向へパディングした断面上で **pyGIMLi** により ERT の順解析を大量実行します。`05_measurements_post_warmup.py` は設定（YAML）を読み込み、内容をCLI引数に変換して `measurements_post_warmup.py` を実行します。`measurements_post_warmup.py` が実際の設計列挙と forward 計算・特徴量作成・NPZ出力を担当します。

**処理概要:**  
1. **YAML読込→CLI起動（`05_measurements_post_warmup.py`）**  
   - 主要キー（`pca, Z, out, n_fields, nz_full/nx_full, n_elec/dx_elec/margin, pattern, n_active_elec, active_policy, active_indices, noise_rel, jobs` など）をCLIフラグにマッピング。  
   - `pattern` を正規化（`wenner-alpha / schlumberger / dipole-dipole / gradient / all`）。未指定時は `wenner-alpha`。  
   - `active_policy=explicit` のとき、`active_indices` の長さ≒`n_active_elec` を検証。  
   - `--dry` でコマンドの表示のみも可。

2. **データセット生成（`measurements_post_warmup.py`）**  
   - **PCA pack**（`mean, components, nz, nx`）と **Z**（N×k）を読込。`Z @ components + mean` で浅部 log₁₀ρ を再構成 → 最下行を繰り返して `nz_full` までパディング。  
   - 地表に **等間隔の固定センサー線**（`n_elec, dx_elec, margin`）を配置。必要に応じて **アクティブ電極サブセット**（`every_other / first_k / explicit`）を選定。  
   - **配列パターン**を列挙：`wenner-alpha / schlumberger / dipole-dipole / gradient / all（重複除去）`。  
   - メッシュ生成 → 画像をセルへサンプリング → **pyGIMLi ERT**（2D/軽量2.5D）で **rhoa** を計算（必要に応じて相対ガウスノイズを付与）。  
   - **設計特徴量** `D=[dAB,dMN,mAB,mMN]` と **正規化特徴量** `Dnorm`（内側幅基準）を作成。ターゲットは **`y=log10(rhoa)`**。  
   - 全フィールドを連結し、学習用の **「設計→応答」テーブル**を1つのNPZに集約。

3. **出力（NPZ・付随ファイル）**  
   - `ert_surrogate_all.npz`：  
     - `Z` (rows, k_lat) … 設計ごとに繰り返した潜在ベクトル  
     - `D` / `Dnorm` (rows, 4) … 設計特徴量とその正規化版  
     - `ABMN` (rows, 4) int32 … 固定線に対する **グローバル0始** 電極インデックス（正準順）  
     - `y` (rows,) … `log10(rhoa)`、`rhoa` (rows,) … ρa[Ω·m]、`k` (rows,) … 幾何学係数  
     - `xs` (n_elec,) … 電極のx座標、`meta` … 実行設定・寸法など（JSON文字列）  
   - `field_source_idx.npy`：Zスライス内の元フィールド行インデックス（再現性用）

**入出力:**  
- **入力:**  
  - PCA pack（`pca_joint.joblib` 等：`mean, components, nz, nx`）  
  - 潜在行列 `Z`（.npy/.npz、形状 N×k）  
  - YAML（実行設定：幾何・配列・サブセット選定・ノイズ・並列など）  
- **出力:**  
  - `outputs/ert_surrogate_all.npz`（上記の設計・応答一式）  
  - `outputs/field_source_idx.npy`

**補足:**    
- `pattern=all` は複数配列のユニオンを **相反則（(A,B,M,N) ~ (M,N,A,B)）の正準化**で重複排除して出力。


### **06_sequential_GPR.py & sequential_GPR.py — ガウス過程回帰（GPR）による逐次測定設計ステップ**
**目的:**  
既存のウォームアップデータを基に、**ガウス過程回帰（GPR）**を用いて ERT（Electrical Resistivity Tomography）の**次に測定すべきABMN**を逐次的に最適化します。`06_sequential_GPR.py` は設定（YAML）を読み込み、各フィールド（Z行）について `sequential_GPR.py` の `run_from_cfg()` を順に実行します。`sequential_GPR.py` は1つのフィールドに対して、**ウォームアップ学習 → アクティブ設計ループ → ログ出力 → 結果の統合**を行います。

---

**処理概要:**  
1. **設定読込とフィールド選択（`06_sequential_GPR.py`）**  
   - YAML設定を読み込み、パラメータ（例: `fields`, `fields_from_medoids`, `workers`）を整理。  
   - フィールドの指定は以下の2通り:
     - `fields_from_medoids`: `medoids_index.npz` の `src_index` を読み込み、自動選定。  
     - `fields`: 明示的にフィールド番号を指定。  
   - 各フィールドに対して `run_from_cfg(cfg, field_index)` を呼び出し、結果をフィールド別フォルダ（`field000/` など）に保存。  
   - 実行後、全フィールドのログを集約し、`GPR_bundle.npz` に統合。

2. **GPRによる逐次設計（`sequential_GPR.py`）**  
   - **データ読込:**  
     PCA潜在ベクトル `Z`、対応するウォームアップデータ (`warmup_npz`)、全応答データ (`y_dataset`) を読み込み、対象フィールドの行を抽出。  
   - **設計空間の構築:**  
     `build.design` 内の **`ERTDesignSpace`** クラスを用いて、Wenner / Schlumberger / Dipole–Dipole / Gradient 配列の ABMN 組み合わせを列挙し、相反則を考慮して重複を除去した**正準化された設計空間**を作成。
   - **特徴量化:**  
     各ABMNを `[dAB, dMN, mAB, mMN]` に変換（距離と位置を分離）、さらに設計空間の**メトリック変換**（例: `cdf+whiten`）を適用して正規化。  
   - **ウォームアップ段階:**  
     - 最初の `warmup_steps` サンプルで **GPRモデル**を学習。  
     - カーネルは「距離」と「位置」それぞれに対する **RBFカーネル**の和または積（`kernel_compose`）＋**WhiteKernel** ノイズ項。  
     - 各ステップでカーネルパラメータ・LML・誤差（RMSE, MAE, R²）を記録。  
   - **アクティブループ:**  
     - 残りの候補設計からランダムに `random_candidates` を抽出。  
     - それぞれの後方平均・分散を求め、指定した**取得関数**（LCB / UCB / EI / MAXVAR / MI）に基づいて最も情報量の高い設計を選択。  
     - 選択した設計の応答値を `y_dataset` から取得し、学習データを更新。  
     - GPRモデルを再学習して次のステップへ進む。  

3. **出力とログ構造:**  
   各フィールドごとに以下のファイルを出力:  
   - `candidate_stats.csv` ：各ステップの候補統計・取得関数値・選択設計  
   - `gpr_params.csv`   ：カーネルパラメータ・ノイズレベル・学習誤差  
   - `seq_log.npz`    ：全ステップの選択履歴・特徴量・応答履歴  
   - `config_used.json`  ：実行時設定の保存  
   実行後、全フィールドの結果を統合:  
   - `bundle_candidate_stats.csv`  
   - `bundle_gpr_params.csv`  
   - `GPR_bundle.npz`  

---

**入出力:**  
- **入力:**  
  - `pca_joblib` : PCAメタファイル（潜在次元情報を含む）  
  - `Z_path`   : 潜在ベクトル（各フィールドの特徴表現）  
  - `warmup_npz` : ウォームアップ測定データ（ABMN, y）  
  - `y_dataset`  : 全設計候補に対する応答（ABMN, yまたはρa）  
  - YAML設定ファイル 例: `configs/sequential_GPR.yml`  
- **出力:**  
  - 各フィールドフォルダ: `candidate_stats.csv`, `gpr_params.csv`, `seq_log.npz`, `config_used.json`  
  - 集約ファイル: `bundle_candidate_stats.csv`, `bundle_gpr_params.csv`, `GPR_bundle.npz`

---

**`build` パッケージと `design.py` の役割**

`sequential_GPR.py` は次のようにインポートします:
from build.design import ERTDesignSpace
**build はフォルダー（Pythonパッケージ）**で、その中の `design.py` に `ERTDesignSpace` が定義されています。`ERTDesignSpace` は GPR スクリプトにおける「測定設計空間の生成と幾何処理」を担うクラスです。

**build/design.py の主な機能:**

- **ABMN設計の列挙:**  
  電極数・最小間隔・重複禁止条件を満たす (A,B,M,N) を全列挙し、相反則 (A,B,M,N) ≡ (M,N,A,B) で重複を削除。

- **連続埋め込み:**  
  各離散設計を正規化連続特徴 `[dAB, dMN, mAB, mMN] ∈ [0,1]^4` に変換。

- **メトリック変換:**  
  設計特徴を統計的に標準化・装飾化する手法を実装（`identity`, `perdim`, `whiten`, `cdf`, `cdf+whiten`）。

- **最近傍探索:**  
  変換後空間で `nearest_1` / `nearest_k` 検索（SciPy cKDTree / sklearn KDTree / ブルートフォース）。

- **ポリシー補助:**  
  `map_uv_to_embed(u ∈ (0,1)^4)` により、ニューラルポリシー等の出力を実現可能な設計空間内の連続 embed へ安全に写像。

以上を踏まえ、`sequential_GPR.py` における設計特徴空間の定義・正規化・探索は、`build/design.py` の `ERTDesignSpace` に依存して動作します。

---

### **07_invert_GPR.py & invert_GPR.py — 逐次設計ログの一括 ERT 逆解析ステップ（pyGIMLi）**
**目的:**  
GPR 逐次設計で得られた **ABMN–応答ログ（seq_log / bundle）** を入力として、pyGIMLi により **地中比抵抗分布（ρ[Ω·m]）をフィールドごとに逆解析**します。`scripts/07_invert_GPR.py` は設定ファイル（YAML）を読み込み、内容をコマンドライン引数に変換して `invert_GPR.py` を実行します。`invert_GPR.py` は **入力バンドルの解析 → メッシュ生成 → 逆解析の実行 → 画像/NPZに集約保存** を担当します。

**処理概要:**  
1. **YAML読込（`07_invert_GPR.py`）**  
   - `--config` で指定された YAML を読み込み、入出力パス・幾何条件・並列数などを取得。  
   - 取得したパラメータをコマンドライン引数に変換して `invert_GPR.py` を起動。  

2. **入力解釈（`invert_GPR.py`）**  
   - `npz` 内の **フィールド別キー**（例: `ABMN__field000`, `y__field000`）を自動検出。  
   - `y` が `log10(ρa)` の場合は **10^y で線形 ρa** に復元。ABMN は **0/1 ベースを自動判定**して 0-based に正規化。  

3. **幾何・メッシュ生成**  
   - 電極数 `n_elec` と間隔 `dx_elec`（または `world_Lx`）から **等間隔電極座標**を生成。  
   - 世界矩形と表層近傍を厚めにとる **2D ERT メッシュ**（`nx_full×nz_full` 目安、または `mesh_area` 指定）を作成。  

4. **逆解析（pyGIMLi / ERTManager）**  
   - 測定誤差率 `err`、正則化強度 `lam`、ロバスト化 `robust` を設定して **各フィールドを順次逆解析**。  
   - 先頭フィールドはメインスレッド、残りは `workers` に応じて **並列**処理（集約時にラベルでソートして順序再現）。  

5. **保存・可視化**  
   - 各フィールドの逆解析ベクトル（セル毎の ρ）とメタ情報（セル中心、世界座標、可視化範囲など）を **単一 NPZ に集約**。  
   - 画像は **線形/対数カラースケール**の PNG を保存（既定では先頭フィールドのみ、`images_all: true` で全フィールド）。  

**入出力:**  
- **入力:**  
  - YAML 設定（例: `configs/invert_GPR.yml`）  
  - `npz` バンドル：`seq_log.npz` または `GPR_bundle.npz`（キー例: `ABMN__field###`, `y__field###`）  
- **出力:**  
  - **画像**：`out`（線形スケールPNG）, `out_log`（対数スケールPNG）／`images_all: true` で全フィールド保存  
  - **NPZ バンドル**（例: `inversion_bundle_GPR.npz`）  
    - 逆解析結果：`inv_rho_cells__field####`（セルごとの ρ[Ω·m]）  
    - 観測再保存：`abmn__field####`, `rhoa__field####`（線形 ρa）  
    - メッシュ/座標：`cell_centers`, `L_world`, `Lz`, `world_xmin/xmax/zmin/zmax` ほか  
    - 可視化範囲：`cmin__field####`, `cmax__field####`  

---

### **08_forward_invert_Wenner.py & forward_invert_Wenner.py — ERT 順解析＋逆解析（Wenner配列）ステップ**
**目的:**  
PCA の潜在ベクトル **Z** から **log₁₀ ρ** フィールドを再構成し、pyGIMLi で **ERT の順解析（ρa シミュレーション）**を実行します。さらに、必要に応じて **逆解析（比抵抗分布の推定）**を行い、画像と NPZ バンドルに集約保存します。`08_forward_invert_Wenner.py` は **設定ファイル（YAML）を読み込み → キーを CLI フラグに変換 → 指定スクリプトを起動**するランナーです（既定は `build/forward_invert_Wenner.py`）。`forward_invert_Wenner.py` は **PCA 再構成 → メッシュ生成 → 配列設計（Wenner/ランダム） → 順解析 →（任意）逆解析 → 出力** を担います。

---

**処理概要:**  
1. **YAML 読込と CLI 変換（`08_forward_invert_Wenner.py`）**  
   - YAML（`inputs/selection/geom/design/forward/output/misc`）の各キーを `KEYMAP` に従って **コマンドライン引数へ写像**。未定義キーは無視。  
   - 変換後、指定ターゲット（既定 `./build/forward_invert_Wenner.py`）を **サブプロセスで起動**。開始/終了時刻と経過秒を表示。

2. **PCA 再構成とフィールド選択（`forward_invert_Wenner.py`）**  
   - `pca`（joblib）から **mean/components/nz/nx** を取得し、`Z` から **log₁₀ρ** を再構成。必要に応じて深さ方向をパディング（`nz_full`）。  
   - フィールドは **明示指定（`--fields`）／範囲（`--n-fields`, `--field-offset`）／seq_log NPZ からの自動抽出**のいずれかで選択。

3. **幾何・メッシュ**  
   - 電極数 `n_elec`、間隔 `dx_elec`（>0 なら **world 長 = 2×margin + (n_elec-1)×dx**）または `world_Lx` から **等間隔の表面電極**を生成。  
   - 表層直下に補助ノードを追加した **2D メッシュ**を作成（`mesh_area`/`quality` 指定可）。縦方向サイズ `Lz` は格子分割から決定。

4. **配列設計（Wenner／ランダム）**  
   - **Wenner**：`a∈[a_min,a_max]` の **A–M–N–B** 列挙で有効 quadruple を全生成。  
   - **ランダム**：幾何制約（`dAB/dMN` 範囲、A/B と M/N の非重複等）下で **一意な AB と MN** をサンプリング。  
   - いずれも **[dAB, dMN, mAB, mMN]** を計算し、**内側幅 L で正規化**した `Dnorm` を構築。  

5. **順解析（pyGIMLi）**  
   - メッシュのセル比抵抗へ `10**(log₁₀ρ)` を割り当て、`ert.simulate()` で **ρa** を計算。`noise_rel` に応じて **相対ガウス雑音**を付与。  
   - 出力バンドル `ert_surrogate.npz` に **Z, D, Dnorm, ABMN, y=log₁₀ρa, ρa, 幾何, メタ** を保存。 

6. **（任意）逆解析**  
   - `--invert` が有効なら、`ERTManager.invert()`（例：`err=0.03, lam=20, robust=True`）で **フィールドごとに逆解析**。  
   - **線形／対数カラー**の PNG を保存（既定は先頭フィールド、`--inv-save-all-png` で全フィールド）。  
   - さらに **inversion bundle**（`inv_log/ inv_rho` 各フィールド、`cell_centers`、世界座標、各フィールドの `abmn`/`rhoa` など）を **単一 NPZ** に集約。


**入出力:**  
- **入力:**  
  - **YAML 設定**（例：`configs/forward_invert_Wenner.yml`）— ランナーが読み込み、CLI へ変換。
  - **PCA**（joblib: mean/components/nz/nx）、**Z**（`.npy`/`.npz`）、（任意）**seq_log NPZ**（フィールド抽出用）。
- **出力:**  
  - **サロゲート統合データ**：`out/ert_surrogate.npz`（`Z, D, Dnorm, ABMN, y, rhoa, k, xs, field_ids, meta`）。:contentReference[oaicite:10]{index=10}  
  - **画像**：`inversion_fieldXXX.png` / `_log.png`（オプション）。
  - **逆解析統合データ**：`inversion_bundle_Wenner.npz`（各フィールドの `inv_rho_cells__field###`, `inv_log_cells__field###` とメタ）。

**補足:**  
逆解析（pyGIMLiによる比抵抗再構成）は **任意設定** ですが、後続のステップで **GPR の結果（予測比抵抗マップ）との比較・評価** に用いられるため、本ステージでは **`invert: true`（または `--invert`）を有効にすることを推奨**します。  

---

### **09_evaluate_GPR_vs_Wenner.py & evaluate_GPR_vs_Wenner.py — GPR と Wenner 配列の評価ステップ（指標・可視化）**
**目的:**  
前段で得られた **逆解析（pyGIMLi）結果** と、PCA から再構成した **真値（log₁₀ρ）** を用いて、**GPR 逐次設計**と**Wenner 配列を用いた基準ケース**を多面的に比較・評価します。`scripts/09_evaluate_GPR_vs_Wenner.py` は **設定ファイル（YAML）を受け取り、対象スクリプトを `--config` 付きで起動**するランナーです。
`evaluate_GPR_vs_Wenner.py` は **評価ロジック本体**で、スカラー指標（MAE/RMSE/相関など）と空間指標（フーリエ相関・形態学的 IoU・JSD）を計算し、プロットと集計を出力します。

**処理概要:**  
1. **YAML 読込・実行（`09_evaluate_GPR_vs_Wenner.py`）**  
   - `--config`（必須）と `--script`（任意、既定 `build/evaluate_GPR_vs_Wenner.py`）を受け取り、**両パスの存在を検証**。  
   - Python サブプロセスで **`<script> --config <yaml>` を実行**し、開始・終了時刻と経過秒を表示します。

2. **評価本体（`evaluate_GPR_vs_Wenner.py`）**  
   - **PCA 真値再構成:** `pca.joblib` と `Z.npz` から、指定フィールドの **真の log₁₀ρ 2D マップ**を再構成。  
   - **入力モード:**  
     - *モードA: 統合NPZ（bundle）比較* — **Wenner 統合NPZ** と **GPR統合NPZ** を同一フィールド集合で比較。各統合NPZは `inv_rho_cells__fieldNNN` と共通メタ（`cell_centers`, `L_world`, `Lz`, `world_xmin`, `world_zmax`）を含むこと。 
     - *モードB: 単体NPZ比較* — 特定フィールドに対する **単一フィールドNPZ** を複数本比較（ラベル付け可）。
   - **指標計算:**  
     - スカラー指標（**log領域**: MAE/RMSE/bias/相関、**線形領域**: MAE/RMSE/相対%）を **深さ重み w(z)=exp(-depth/λ)**（任意）付きで算出。
     - 空間類似度：**2Dフーリエ振幅スペクトル相関**, **形態学的IoU（Otsu二値化＋クロージング）**, **Jensen–Shannonダイバージェンス**。
   - **サブセット評価:** 全セル（all）、**bottom25% / top25%**（真値の四分位）、**shallow50% / deep50%**（物理的深さ）で並列に算出。
   - **可視化:** パリティ図、残差ヒストグラム、残差 vs 真値の散布図を PNG として出力（任意）。

**入出力:**  
- **入力（共通）:**  
  - `pca`: PCA メタ（`mean`, `components`, `nz`, `nx` を含む joblib）  
  - `Z`: PCA 係数（`Z` 配列を含む npz）  
  - `out_dir`: 出力先ディレクトリ  
  - `lambda_depth`（任意）: 深さ重みの減衰長（m, 0以下で無効）  
  - `plots`（任意）: 可視化 PNG を出力するか  
  - `scatter_max`（任意）: 散布図の点数上限（ダウンサンプル）  
- **入力（モードA: 統合NPZ比較）:**  
  - `wenner_bundle`: Wenner **統合NPZファイル**  
  - `GPR_bundle`（または `inv_npz_bundle` / `inv_npz`）: 比較対象（例: **GPR 統合NPZ**）  
  - `fields`（任意）: 比較対象とするフィールド番号のリスト（0始まり）
- **入力（モードB: 単体NPZ）:**  
  - `mode: "standalone"`  
  - `inv_npz`: 単体NPZのパス配列（各 NPZ は `inv_rho_cells`, `cell_centers`, `L_world`, `Lz`, `world_xmin`, `world_zmax` を含む）  
  - `field_idx`: PCA 真値再構成に用いるフィールド番号  
  - `labels`（任意）: グラフや出力に用いる名前配列（`inv_npz` と同数）
- **出力:**  
  - `summary_metrics.json / .csv`：**ALL/深浅/四分位サブセット**ごとの全指標を集約  
  - `parity_*.png`, `residual_hist_*.png`, `residual_vs_true_*.png`（`plots: true` の場合）  
  - `per_cell_log10_*.csv`（`write_per_cell: true` の場合）

---

### **10_plot_summary.py — 評価サマリーの可視化・集計ステップ**
**目的:**  
`summary_metrics.csv`（各フィールド×各手法の評価指標を集約した表）を読み込み、**サブセット別**（`top25 / all / bottom25 / shallow50 / deep50`）に (1) GPR vs WENNER の**分布比較**（箱ひげ図）、(2) GPR 相対改善率の**横棒グラフ**、(3) **IQR（Interquartile Range）/相対IQR**の比較、(4) **生データ/統計/ペア表**の CSV を自動生成します。

**処理概要:**  
1. **CSV 読込と前処理**  
   - `label, subset, source` と主要指標列（`mae_log10, rmse_log10, ...`）を検証し、不足列は `NaN` で補完。
2. **分布比較（GPR vs WENNER）**  
   - 各指標について、**GPR 左・WENNER 右**の並列箱ひげ図を作成（平均線付き）。
3. **相対改善率プロット**  
   - 指標ごとに **2種類の%改善**を計算・可視化：  
     - **PF**: フィールド別相対変化の平均 `mean_label(100×(GPR−WENNER)/|WENNER|)`  
     - **OVR**: 平均値どうしの相対変化 `100×(mean(GPR)−mean(WENNER))/|mean(WENNER)|`  
   - 相関/IoU は**プラスが改善**, 誤差/JSD は**マイナスが改善**として緑/赤で塗分け、PF（斜線）と OVR（ドット）を**ハッチで区別**。対応する CSV も保存。
4. **スプレッド/IQR**  
   - `IQR = Q3−Q1` と **Relative IQR(%) = 100×IQR/|median|** を算出し、GPR vs WENNER を**棒グラフ**で比較。CSV も出力。 
5. **値のエクスポート**  
   - サブセットごとに以下を保存：  
     - `values_<subset>.csv`：**元の値**  
     - `values_stats_<subset>.csv`：source×metric の**集計統計**（count/mean/std/min/median/max/Q1/Q3）  
     - `values_paired_<subset>.csv`：**同一 label の WENNER/GPR/差分**のペア表（平均で重複解消）。

**入出力:**  
- **入力:**  
  - `--csv`：評価サマリー CSV（既定 `data/evaluation/summary_metrics.csv`）。必須列：`label, subset, source` と主要指標（不足時は自動補完）。
- **出力（サブセットごとに `--outdir` 配下へ）:**  
  - 可視化：`compare_<subset>__*.png`（箱ひげ図）, `relative_change_<subset>.png`（相対%）  
  - IQR：`iqr_<subset>.png`, `relative_iqr_<subset>.png`  
  - CSV：`values_*.csv`, `values_stats_*.csv`, `values_paired_*.csv`, `relative_change_*.csv`, `relative_change_summary_*.csv`, `iqr_*.csv` ほか

---


## 📘 背景説明（Background）

### 電気比抵抗トモグラフィ（Electrical Resistivity Tomography, ERT）
**電気比抵抗トモグラフィ（ERT）** は、地下の構造や水分状態を「電気の通りやすさ（比抵抗）」から推定する**地球物理学的な可視化手法**です。  
地表やボーリング孔に多数の電極を設置し、**一部の電極（A, B）から電流を注入**し、**別の電極（M, N）間で電位差を測定**します。  
これをさまざまな電極組み合わせで繰り返すことで、地下の**電気比抵抗分布（ρ）**を推定するための情報が得られます。  
比抵抗は**含水率・塩分濃度・土壌や岩石の種類**などに依存するため、ERTを用いれば**地下水の浸潤、塩水侵入、汚染拡散、地盤構造の把握**などを非破壊的に行うことができます。

---

### 電極配列（Electrode Arrays）とは
電極配列とは、**複数の電極（例：32本）**を地表に設置し、そこから電流を流す電極ペア（A–B）と電位を測る電極ペア（M–N）を**組み合わせて測定を繰り返すための設計ルール**です。  

たとえば32本の電極を用いる場合、選び得る電極の組み合わせ（A–B–M–N）は非常に多数存在し、これらをどの順序・どの間隔で選ぶかによって、測定の深さ・感度分布・ノイズ特性が大きく変化します。  
この**電極組み合わせの列挙方法（組み合わせの体系）**こそが「電極配列」と呼ばれるもので、代表的な配列には以下のようなものがあります：

| 配列名 | 特徴 | 主な用途 |
|:--|:--|:--|
| **Wenner 配列** | 4電極（A–M–N–B）が等間隔で配置される。組み合わせの生成が単純。 | ノイズに強く、浅部構造の解析に適する |
| **Schlumberger 配列** | 電流電極（A, B）を広く、電位電極（M, N）を近く配置 | 深部構造に感度が高い |
| **Dipole–Dipole 配列** | 電流ペアと電位ペアを離して配置 | 空間分解能が高いがノイズの影響を受けやすい |
| **Gradient 配列** | 電流電極を固定し、多数の電位電極で同時計測 | 測定効率が高く、データ密度が高い |

本リポジトリでは、**Wenner–alpha 配列**を基準配列として採用しています。  
これは感度分布が対称で、測定設計の最適化手法（GPR逐次設計）との比較基準として扱いやすいためです。

---

### 逆解析（Inversion）とは
ERT測定で得られるのは、各電極組み合わせに対応する**地表での電位差（観測データ）**です。  
しかし、私たちが知りたいのは地下の**比抵抗分布**そのものです。  
そこで、観測データを最もよく説明する地下モデルを数値的に求める必要があります。  
この計算過程を**逆解析（inversion）**と呼びます。
本リポジトリでは、オープンソースの物理探査ライブラリ **pyGIMLi** を用いて逆解析を行い、**逐次的に選ばれた測定系列から再構成された地下比抵抗分布**を得ます。
この逆解析結果は、後述の**GPR逐次設計による測定最適化の性能評価**に用いられます。

---

### GPRによる逐次実験設計（Sequential Design via Gaussian Process Regression）
**ガウス過程回帰（GPR）** は、観測点間の**統計的な相関関係**をモデル化する手法です。  
ERTのような測定データに対して、GPRを用いることで、すでに観測した電極組み合わせの結果から**未観測の組み合わせ（A–B–M–N）の測定値を確率的に推定**し、「次にどの組み合わせを測ると最も情報量が増えるか」を選ぶことができます。
本リポジトリにおけるGPR逐次設計は、**逆解析後の地下構造ではなく、逆解析前の「測定データ空間（電位差データ）」に対して適用される近似的な最適化**です。すなわち、GPRは電極組み合わせパターン間の相関を学習し、**測定効率を最大化する測定系列**を逐次的に生成します。そして、その結果の有効性は、GPRで選ばれた測定系列をもとに逆解析を行い、得られた**再構成後の地下比抵抗分布の精度（Wenner基準との比較）**によって評価されます。  

---

### 🧩 まとめ
- **電気比抵抗トモグラフィ（ERT）** は、電流注入と電位差測定を多くの電極組み合わせで行い、地下の比抵抗分布を推定する手法である。  
- **電極配列（Electrode Array）** は、複数の電極（例：32本）から多数の電極ペアを列挙して測定を繰り返すための設計方式であり、測定の感度や深度特性を決定する。  
- **逆解析（Inversion）** は、これらの測定データから地下の比抵抗分布を再構成する計算過程であり、GPR逐次設計の結果評価に不可欠なステップである。  
- **本リポジトリにおけるGPRによる逐次設計** は、**逆解析前の測定データ空間**に対して適用される近似的な最適化であり、その有効性は、**逆解析後に再構成された地下比抵抗分布を通して定量的に評価される**。  
