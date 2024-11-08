import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import easyocr
import pandas as pd
import numpy as np
from io import BytesIO
import cv2
import copy

def create_pos_data(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return {'x': x, 'y': y, 'width': w, 'height': h}

def display_np(img):
    check=Image.fromarray(img)
    st.image(check, caption="チェック画像")

# 言語選択のオプション
language_options = ["日本語", "英語"]
# 選択に応じて言語コードをマッピング
language_code_map = {
    "日本語": "ja",
    "英語": "en"
}

# 画像の最大幅と高さを設定
max_width, max_height = 800, 800  # 必要に応じて変更
def_num_columns=2
def_num_offset=0

if 'num_columns' not in st.session_state.keys():
    st.session_state['num_columns'] = def_num_columns
if 'num_offset' not in st.session_state.keys():
    st.session_state['num_offset'] = def_num_offset
if 'result' not in st.session_state.keys():
    st.session_state['result'] = []

# ページ幅の設定
st.set_page_config(layout="wide")

# アプリのタイトル
st.title("TABLE OCR")


# 画像アップロード
uploaded_image = st.sidebar.file_uploader("画像のアップロード", type=['png', 'jpg', 'jpeg'])

# 言語選択マルチセレクト
selected_languages = st.sidebar.multiselect("言語を選択", language_options,"日本語")
# 選択された言語のコードを格納
selected_codes = [language_code_map[lang] for lang in selected_languages]


if uploaded_image:
    # 画像の読み込み
    image = Image.open(uploaded_image)
    
    # 行や列の数が決まっている場合はそれに合わせて調整してください。
    # ここでは簡易的に列数を3として処理
    st.session_state['num_columns'] = st.sidebar.number_input("表の列数を入力", min_value=1, value=def_num_columns)
    st.session_state['num_offset'] = st.sidebar.number_input("オフセット値を入力", min_value=0, value=def_num_offset)

    #OCR実行ボタンの配置
    def do_ocr():
        # EasyOCR Readerのインスタンスを作成
        # reader = easyocr.Reader(['ja', 'en'])  # 日本語と英語対応
        reader = easyocr.Reader(selected_codes,model_storage_directory=".EasyOCR/model/")
        # OCRを実行し、テキスト情報を抽出
        st.session_state['result'] = reader.readtext(np.array(cropped_image), detail=1)
        
    button=st.sidebar.button("OCR実行",on_click=do_ocr)
    
    # st_cropperで画像のトリミング
    st.write("OCR実行エリアの選択")
    cropped_image = st_cropper(image, realtime_update=True, box_color='blue', aspect_ratio=None)

    # 画像の幅と高さを取得
    image_width, image_height = cropped_image.size
    
    # 画像が最大幅または高さを超える場合はリサイズ
    if image_width > max_width or image_height > max_height:
        # アスペクト比を保持しつつリサイズ
        cropped_image.thumbnail((max_width, max_height), Image.BICUBIC)

    with st.sidebar.expander("検討中の機能"):
        edge_threshold = st.slider("エッジ検出のための閾値",-500,500,100,5)
        morph_size=st.slider("膨張のための値",1,10,1,1)
        # cropped_image=np.array(cropped_image)
        # cropped_image=cv2.imread(uploaded_image)
        _cropped_image=np.array(cropped_image)
        height, width, channels = _cropped_image.shape
        # グレースケール変換と二値化処理
        gray_img = cv2.cvtColor(_cropped_image, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        #エッジ検出
        edges = cv2.Canny(gray_img, edge_threshold, edge_threshold)
        #膨張処理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(morph_size,morph_size))
        dilates = cv2.dilate(edges, kernel)
        # 輪郭の検出
        contours, hierarchy = cv2.findContours(dilates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # エクスポート用イメージ配列を作成しておく
        export_img = copy.deepcopy(_cropped_image)
        # JSON用の辞書型配列
        export_array = {
            'width': width,
            'height': height,
            'results': []
        }
        for i in range(len(contours)):
            # 色を指定する
            color = np.random.randint(0, 255, 3).tolist()   
            if cv2.contourArea(contours[i]) < 3000:
                continue  # 面積が小さいものは除く
            # 階層が第１じゃなかったら ... 
            if hierarchy[0][i][1] != -1:
                # 配列に追加
                export_array['results'].append(create_pos_data(contours[i]))
                # 画像に当該の枠線を追加
                cv2.drawContours(export_img, contours, i, color, 3)
        display_np(export_img)

    # # 左右に分けるレイアウト作成
    left_col, right_col = st.columns(2,vertical_alignment="center")

    # 左側：トリミングエリア
    with left_col:
        # トリミングされた画像を表示
        st.image(cropped_image, caption="OCR実行エリア")


    with right_col:
        
        # OCRの結果からセル内容をリストに格納
        data = []
        for i in range(st.session_state['num_offset']):
            data.append("---")

        for (bbox, text, prob) in st.session_state['result']:
            data.append(text)
                    
        # OCR結果を表形式に整理
        table_data = [data[i:i + st.session_state['num_columns']] for i in range(0, len(data), st.session_state['num_columns'])]
        df = pd.DataFrame(table_data)

        # 表を表示
        st.write("OCR結果")
        st.dataframe(df,use_container_width=True)

        # CSVとしてダウンロード
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="CSVとしてダウンロード",
            data=csv,
            file_name='table_data.csv',
            mime='text/csv',
        )