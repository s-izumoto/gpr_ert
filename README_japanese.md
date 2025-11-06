# GPR–ERT 逐次設計と評価パイプライン

このリポジトリは、**電気比抵抗トモグラフィ（ERT）** と **ガウス過程回帰（GPR）** に基づく **逐次実験設計** の完全なワークフローを実装しています。  
不均一な比抵抗場の生成から、逆解析、そしてWenner配列を用いた基準との定量的比較までのすべての段階をカバーします。

# GPRベース逐次設計：性能概要

**Wenner配列を用いた基準と比較**して、**Mutual Information ベースの獲得関数**を用いた GPR 逐次設計 は、全体的な評価指標において一貫した性能向上を示しました。比較画像の例は `images_example/` フォルダーにあります。

- **大津のしきい値処理および形態学的フィルタリング後のIoU** は平均で **約8% 向上**  
- **RMSE** は **6% 改善**  
- **Pearson r** および **Spearman ρ** はそれぞれ **4% 改善**  
- **Jensen–Shannon divergence (JSD)** は **2% 改善**  
- 特に **領域の下半分（深部）** において改善が顕著で、**Pearson r** は **18% 向上**、**Spearman ρ** は **16% 向上**、**RMSE** は **9% 低下** しました。  
- **表層半分** においてもすべての指標で数パーセントの改善が見られ、**悪化した指標はありません** でした。

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

# 3) バージョン確認
python -V
pip -V
```

---
## 🧭 プロジェクト構造

このリポジトリは、**設定・実行・コア機能**を明確に分離した階層構造を持っています。\
このモジュール化された構成により、パイプラインの**再現・デバッグ・拡張**が容易になります。

    gpr_ert/
    ├── build/                       # 再利用可能なモジュール群とコアロジック
    │   ├── __init__.py
    │   ├── make_fields.py           # 合成2次元比抵抗マップを生成
    │   ├── fit_pca.py               # PCAを学習し、比抵抗マップを低次元空間に圧縮
    │   ├── cluster_pca.py           # PCA空間でクラスタリングを行い代表サンプルを選択
    │   ├── design.py                # ERT設計空間の列挙（ABMN生成、特徴量作成）
    │   ├── measurements_warmup.py   # 初期（ウォームアップ）Wenner測定の順計算
    │   ├── measurements_post_warmup.py  # 拡張配列型の順計算
    │   ├── gpr_seq_core.py          # ガウス過程回帰（GPR）のコア実装
    │   ├── sequential_GPR.py        # gpr_seq_coreをラップする逐次設計ドライバ
    │   ├── invert_GPR.py            # GPR選択測定データの一括逆解析
    │   ├── forward_invert_Wenner.py # Wenner配列基準の順計算＋逆解析ワークフロー
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
    │   ├── forward_invert_Wenner.yml# Wenner配列基準の順計算＋逆解析設定
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

この構成により、**ロジック・実行・設定の明確な分離**が実現し、\
透明性と再現性の高い研究ワークフローが可能になります。

------------------------------------------------------------------------

## パイプラインの概要

ワークフローは**10個のモジュール化されたスクリプト**から構成され、\
それぞれが前段の出力を次の入力として受け取ります。

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ステップ                   スクリプト                         目的
  -------------------------- ---------------------------------- ------------------------------------------------------------------------------------------------------
  **01**                     `01_make_fields.py`                多様な地質シナリオを模擬した合成2次元比抵抗フィールドを生成。

  **02**                     `02_fit_pca.py`                    PCAモデルを学習し、比抵抗マップを低次元潜在空間に圧縮・再構成。

  **03**                     `03_cluster_pca.py`                潜在ベクトルをクラスタリングし、代表サンプルを抽出。

  **04**                     `04_measurements_warmup.py`        Wenner-α配列を用いたERT測定をシミュレーションし、GPR初期化用の**ウォームアップデータセット**を構築。

  **05**                     `05_measurements_post_warmup.py`   Schlumberger、Dipole-Dipole、Gradientなど複数の配列型から拡張データを生成。

  **06**                     `06_sequential_GPR.py`             **ガウス過程回帰（GPR）**を適用。ウォームアップデータで初期学習後、獲得関数（UCB, LCB, EI, MI,
                                                                MAXVAR）を用いて逐次的に新しい測定を選択。

  **07**                     `07_invert_GPR.py`                 逐次GPRによる測定データを逆解析し、地下比抵抗分布を再構築。

  **08**                     `08_forward_invert_Wenner.py`      Wenner配列基準の順計算＋逆解析を行い、参照データを作成。

  **09**                     `09_evaluate_GPR_vs_Wenner.py`     GPRベースとWennerベースの逆解析結果を、誤差・空間的類似度などで定量評価。

  **10**                     `10_plot_summary.py`               評価指標をまとめ、ボックスプロットや相対改善率などを可視化。


------------------------------------------------------------------------

## ウォームアップフェーズと逐次GPR

**GPRのアクティブフェーズ開始前**に、安定化のための**ウォームアップ期間**を設けます。

-   この期間では、**固定された測定構成（通常はWenner-α配列）**を用いてシミュレーションを実施します。\
-   得られたデータが**ウォームアップデータセット**となり、GPRモデルの初期訓練データになります。\
-   その後、**逐次GPRフェーズ**が始まり、GPRモデルが候補測定点の不確実性を推定し、\
    **獲得関数**（例：相互情報量 Mutual Information や 期待改善 Expected
    Improvement）を適用して\
    次にサンプリングすべき構成を選びます。\
-   新しい測定データを訓練集合に追加し、GPRを再学習して精度を向上させていきます。

この戦略により、**探索（不確実性の低減）**と**活用（高影響領域の重点測定）**を両立させ、\
ウォームアップからアクティブフェーズへ自然に接続します。

------------------------------------------------------------------------

## 概念的な流れ

1.  **合成フィールド生成:**
    現実的な地質的不均質性を再現する比抵抗フィールドを作成。\
2.  **次元削減:** PCAにより比抵抗マップを低次元潜在空間へ写像。\
3.  **サンプル選択:** シミュレーション対象となる代表ケースを選択。\
4.  **ウォームアップ順計算:** 標準配列構成を用いて初期ERT応答を収集。\
5.  **候補拡張:** 多数の配列構成に対する応答をシミュレーション。\
6.  **逐次GPR:**
    ウォームアップ後のGPRモデルに獲得関数を適用し、最適な次測定点を逐次選択。\
7.  **逆解析:** 逐次取得データに基づいて地下比抵抗を再構築。\
8.  **基準逆解析:** Wenner基準で同様の逆解析を実施。\
9.  **評価:** 再構築結果と真値を統計・空間指標で比較。\
10. **可視化:** 評価結果をグラフ化し、最終的な比較図を生成。

------------------------------------------------------------------------

## 主な特徴

-   **ウォームアップ → アクティブGPR** の2段階ワークフロー\
-   **複数配列型対応:** Wenner, Schlumberger, Dipole--Dipole, Gradient\
-   **獲得関数:** UCB, LCB, EI, MAXVAR, Mutual Information\
-   **分離可能カーネル:**
    双極子距離と位置に対して独立したRBFを使用し、長さスケールを自動追跡\
-   **YAML駆動の設定** により完全再現可能な実験\
-   **フィールドごとのログ出力**（CSV,
    NPZ）でカーネルハイパーパラメータ、獲得統計、逆解析を記録

------------------------------------------------------------------------

## 典型的な出力

-   合成および逆解析された比抵抗マップ\
-   PCA成分および潜在表現\
-   ウォームアップ／ポストウォームアップ測定データセット\
-   逐次GPRログおよびカーネル診断\
-   逆解析モデル（GPR vs Wenner）\
-   評価サマリーおよび可視化結果

------------------------------------------------------------------------

## ⚙️ エンドツーエンド・パイプラインワークフロー

このセクションでは、リポジトリに実装された**完全再現可能なワークフロー**をまとめます。\
各ステップは順に実行され、対応する`configs/`内のYAMLファイルを使用します。\
ワークフローは、**合成フィールド生成 →
逆解析・評価・可視化**までを網羅します。

------------------------------------------------------------------------

1.  **合成比抵抗フィールドの生成**\
    多様な地質条件を表す2次元比抵抗マップを作成。

    ``` bash
    python scripts/01_make_fields.py --config configs/make_fields.yml
    ```

2.  **PCAによる潜在空間への射影**\
    比抵抗マップを圧縮し、潜在表現を取得。

    ``` bash
    python scripts/02_fit_pca.py --config configs/fit_pca.yml
    ```

3.  **潜在表現のクラスタリング**\
    PCA後の潜在ベクトルをクラスタリングし、代表サンプルを抽出。

    ``` bash
    python scripts/03_cluster_pca.py --config configs/cluster_pca.yml
    ```

4.  **ウォームアップ測定シミュレーション（Wenner配列）**\
    pyGIMLiを用いてWenner-α配列で順計算を実施し、GPR初期化用データを収集。

    ``` bash
    python scripts/04_measurements_warmup.py --config configs/measurements_warmup.yml
    ```

5.  **拡張測定シミュレーション（複数配列）**\
    Schlumberger, Dipole-Dipole, Gradientなど追加配列で順計算を実施。

    ``` bash
    python scripts/05_measurements_post_warmup.py --config configs/measurements_post_warmup.yml
    ```

6.  **逐次ガウス過程回帰（GPR）**\
    ウォームアップデータで初期学習し、アクティブフェーズで獲得関数を用いて\
    次に最も情報量の高い測定構成を選択。

    ``` bash
    python scripts/06_sequential_GPR.py --config configs/sequential_GPR.yml
    ```

7.  **逐次GPR結果の逆解析**\
    選択された測定データを用いて逆解析を実施し、地下比抵抗モデルを再構築。

    ``` bash
    python scripts/07_invert_GPR.py --config configs/invert_GPR.yml
    ```

8.  **Wenner基準での順計算＋逆解析**\
    Wenner配列を用いた同等の解析を実施し、基準データを作成。

    ``` bash
    python scripts/08_forward_invert_Wenner.py --config configs/forward_invert_Wenner.yml
    ```

9.  **GPRとWennerの比較評価**\
    統計指標・空間的類似度に基づいて両者を定量比較。

    ``` bash
    python scripts/09_evaluate_GPR_vs_Wenner.py --config configs/evaluate_GPR_vs_Wenner.yml
    ```

10. **サマリープロットとレポートの生成**\
    評価結果を可視化し、ボックスプロットや相対改善率などをまとめて出力。

    ```bash     
    python scripts/10_plot_summary.py --config configs/plot_summary.yml`
    ```

---

