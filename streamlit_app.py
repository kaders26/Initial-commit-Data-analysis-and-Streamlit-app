import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import calendar
import pickle


# Sayfa ayarları
st.set_page_config(page_title="Superstore Sales Dashboard", layout="wide")

# Başlık
st.title('📊 Superstore Veri Analizi ve Görselleştirme')

# Açıklama
st.write("""
Bu uygulama, Superstore satış verileri üzerinde keşifsel veri analizi yaparak kullanıcı etkileşimli görselleştirmeler sunmaktadır.
""")

# Veri Yükleme
@st.cache_data
def load_data():
    df = pd.read_csv("archive/Sample - Superstore.csv", encoding='ISO-8859-1')   # Yolu doğru şekilde yazmayı unutma!
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df

df = load_data()

# Modeli yükle
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

model = load_model()


# Sidebar Seçenekleri
options = st.sidebar.radio(
    'Gitmek istediğiniz analiz bölümü:',
    ('Ana Sayfa','Kategori Bazlı Satışlar', 'Bölge Bazlı Satışlar', 'Aylık Satış Trendleri', 
     'İndirim vs Kar İlişkisi', 'Mevsimlere Göre Satışlar', 'ML ile Satış Tahmini')
)

# Ana Sayfa İçeriği
if options == 'Ana Sayfa':
    st.title("📊 Superstore Veri Analizi ve Raporları")

    st.markdown("""
        ## 🎯 Proje Hedefi

Bu proje kapsamında, bir perakende satış veri seti üzerinde detaylı veri analizi, görselleştirme ve temel makine öğrenmesi uygulamaları gerçekleştirilmiştir.  
Amaç, veri içerisindeki trendleri, desenleri ve iş kararlarını destekleyecek önemli içgörüleri ortaya çıkarmaktır.

---

## 🛠️ Kullanılan Adımlar

- **Veri Temizleme (Data Cleaning):**  
Eksik ve aykırı değerlerin tespiti ve yönetimi yapılmıştır. 5000₺ üzerindeki satış değerleri çıkarılarak veri normalleştirilmiştir.

- **Keşifsel Veri Analizi (EDA):**  
Veri setindeki değişkenlerin dağılımları incelenmiş, kategori ve bölge bazlı satış ve kâr analizleri yapılmıştır.

- **Özellik Mühendisliği (Feature Engineering):**  
Satış başına fiyat (`price_per_unit`), kâr marjı (`profit_margin`), sipariş günü (`order_day`) gibi yeni değişkenler oluşturulmuştur.

- **Makine Öğrenmesi (Machine Learning):**  
Random Forest modeli kullanılarak satış tahmini gerçekleştirilmiş ve performansı MAE, RMSE, R² gibi metriklerle değerlendirilmiştir.

- **Veri Görselleştirme (Visualization):**  
Kategori, bölge, sezon ve gün bazında satış ve kâr verileri görselleştirilerek interaktif dashboard oluşturulmuştur.

---

## ✨ Elde Edilen Başlıca İçgörüler

- **Teknoloji** ve **ofis malzemeleri** kategorileri satışta lider konumdadır.
- **Batı** ve **Doğu** bölgeleri satış açısından ön plana çıkmıştır.
- **İndirim oranı** arttıkça **kârlılıkta düşüş** gözlemlenmiştir.
- **Kasım** ve **Aralık** aylarında satışlarda önemli artışlar görülmüştür.
- **Hafta içi** satışlarının hafta sonuna göre daha yüksek olduğu belirlenmiştir.

    """)

    # İlgili analizlerin başlıkları
    st.markdown("""
        ### Hangi analizleri keşfetmek istersiniz?
        - **Kategori Bazlı Satışlar**: Hangi ürün kategorileri daha çok satılıyor?
        - **Bölge Bazlı Satışlar**: Bölgelere göre satış verilerini detaylıca inceleyin.
        - **Aylık Satış Trendleri**: Zaman serisi analizleriyle satışların aylık değişimini gözlemleyin.
        - **İndirim vs Kar İlişkisi**: İndirim oranlarının kar üzerindeki etkisini keşfedin.
        - **Mevsimsel Satışlar**: Satışların mevsimsel trendlere göre nasıl değiştiğini inceleyin.
        - **ML ile Satış Tahmini**: Makine öğrenmesiyle satış tahminleri yaparak geleceği öngörün.
    """)

# Diğer analiz bölümleri burada kontrol edilecek
if options == 'Kategori Bazlı Satışlar':
    # Kategori bazlı satış analizi kodu
    st.subheader('Kategori Bazlı Satışlar')
    # Kategori bazlı analiz kodu burada olacak

elif options == 'Bölge Bazlı Satışlar':
    # Bölge bazlı satış analizi kodu
    st.subheader('Bölge Bazlı Satışlar')
    # Bölge bazlı analiz kodu burada olacak


# Kategori Bazlı Satışlar
if options == 'Kategori Bazlı Satışlar':
    st.header('📦 Kategori Bazlı Satış ve Kar')

    # Kategori bazlı toplam satış, kar, satılan miktar ve indirim hesaplama
    category_summary = df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum', 'Discount': 'mean'}).reset_index()

    # 1. Bar Grafik - Toplam Satış
    st.subheader('Toplam Satış (Bar Grafik)')
    fig1 = px.bar(category_summary, x='Category', y='Sales', title="Kategoriye Göre Toplam Satış", 
                  labels={'Sales': 'Toplam Satış (USD)', 'Category': 'Kategori'})
    st.plotly_chart(fig1)

    # 2. Pie Chart - Satış Dağılımı
    st.subheader('Kategoriye Göre Satış Dağılımı (Pie Chart)')
    fig2 = px.pie(category_summary, names='Category', values='Sales', title="Kategoriye Göre Satış Dağılımı")
    st.plotly_chart(fig2)

    # 3. Scatter Plot - Satış ve Kar Arasındaki İlişki
    st.subheader('Satış ve Kar Arasındaki İlişki (Scatter Plot)')
    fig3 = px.scatter(category_summary, x='Sales', y='Profit', color='Category', 
                      size='Quantity', hover_name='Category', title="Satış ve Kar Arasındaki İlişki")
    st.plotly_chart(fig3)

    # 4. Box Plot - Satışlar ve İndirim Arasındaki İlişki
    st.subheader('Satış ve İndirim Arasındaki İlişki (Box Plot)')
    fig4 = px.box(df, x='Category', y='Sales', color='Category', points='all', 
                  title="Kategoriye Göre Satış ve İndirim Dağılımı")
    st.plotly_chart(fig4)

    # 5. Heatmap - Satış ve Karın Kategoriye Göre Korelasyonu
    st.subheader('Kategoriye Göre Satış ve Kar Korelasyonu (Heatmap)')
    correlation_matrix = category_summary[['Sales', 'Profit']].corr()
    fig5 = px.imshow(correlation_matrix, text_auto=True, title="Satış ve Kar Korelasyonu")
    st.plotly_chart(fig5)

    # 6. Bar Grafik - Kar Marjı
    st.subheader('Kategoriye Göre Kar Marjı (Bar Grafik)')
    category_summary['Profit Margin'] = category_summary['Profit'] / category_summary['Sales'] * 100
    fig6 = px.bar(category_summary, x='Category', y='Profit Margin', title="Kategoriye Göre Kar Marjı (%)",
                  labels={'Profit Margin': 'Kar Marjı (%)', 'Category': 'Kategori'})
    st.plotly_chart(fig6)
    
# Bölge Bazlı Satışlar
elif options == 'Bölge Bazlı Satışlar':
    st.header('🌍 Bölgesel Satış Dağılımı')

    # Bölgeye göre toplam satışları hesaplama
    region_sales = df.groupby('Region')['Sales'].sum().reset_index()

    # 1. Bar Grafiği - Bölgelere Göre Satışlar
    st.subheader('Bölgelere Göre Toplam Satış (Bar Grafik)')
    fig1 = px.bar(region_sales, x='Region', y='Sales', title="Bölgelere Göre Toplam Satış",
                  labels={'Sales': 'Toplam Satış (USD)', 'Region': 'Bölge'}, color='Sales', color_continuous_scale='Viridis')
    st.plotly_chart(fig1)

    # 2. Pie Chart - Bölgelere Göre Satış Dağılımı
    st.subheader('Bölgelere Göre Satış Dağılımı (Pie Chart)')
    fig2 = px.pie(region_sales, names='Region', values='Sales', title="Bölgelere Göre Satış Dağılımı", 
                  color='Region', color_discrete_sequence=sns.color_palette('pastel').as_hex())
    st.plotly_chart(fig2)

    # 3. Bölgelere Göre Aylık Satış Trendleri
    st.subheader('Bölgelere Göre Aylık Satış Trendleri')
    region_monthly_sales = df.groupby([df['Order Date'].dt.to_period('M'), 'Region'])['Sales'].sum().reset_index()
    region_monthly_sales['Order Date'] = region_monthly_sales['Order Date'].astype(str)

    fig3 = px.line(region_monthly_sales, x='Order Date', y='Sales', color='Region', 
                   title="Bölgelere Göre Aylık Satış Trendleri", labels={'Sales': 'Satış (USD)', 'Order Date': 'Tarih'})
    st.plotly_chart(fig3)

    # 4. Bölgelere Göre Satış ve Kar Arasındaki İlişki (Scatter Plot)
    st.subheader('Bölgelere Göre Satış ve Kar Arasındaki İlişki')
    region_profit = df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    
    fig4 = px.scatter(region_profit, x='Sales', y='Profit', color='Region', size='Sales', hover_name='Region', 
                      title="Bölgelere Göre Satış ve Kar İlişkisi", labels={'Sales': 'Toplam Satış (USD)', 'Profit': 'Toplam Kar (USD)'})
    st.plotly_chart(fig4)

    # 5. Satışlar ve Karların Bölgeye Göre Karşılaştırılması
    st.subheader('Bölgelere Göre Satış ve Kar Karşılaştırması (Stacked Bar)')
    region_sales_profit = df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

    fig5 = px.bar(region_sales_profit, x='Region', y=['Sales', 'Profit'], title="Bölgelere Göre Satış ve Kar Karşılaştırması",
                  labels={'value': 'Değer (USD)', 'Region': 'Bölge'}, color_discrete_map={'Sales': 'blue', 'Profit': 'green'})
    st.plotly_chart(fig5)

    # 6. Bölge Seçimi ile Kar ve Satışın Görselleştirilmesi
    selected_region = st.selectbox("Bölge Seçin", region_sales['Region'].unique())
    selected_region_data = df[df['Region'] == selected_region].groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

    st.subheader(f"{selected_region} Bölgesi İçin Kategori Bazlı Satış ve Kar")
    fig6 = px.bar(selected_region_data, x='Category', y='Sales', title=f"{selected_region} Bölgesi - Kategoriye Göre Satış")
    st.plotly_chart(fig6)


elif options == 'Aylık Satış Trendleri':
    st.header('📈 Aylık Satış Trendleri (Gelişmiş)')

    # Order Date'ten ayrı ayrı Yıl ve Ay bilgisi çıkaralım
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month

    # Yıl ve Ay bazında satışları gruplayalım
    monthly_sales = df.groupby(['Year', 'Month']).agg({'Sales': 'sum'}).reset_index()

    # Ay isimlerini ekleyelim
    monthly_sales['Month_Name'] = monthly_sales['Month'].apply(lambda x: calendar.month_name[x])

    # Kullanıcıya yıl seçtirelim
    years = monthly_sales['Year'].unique()
    selected_year = st.selectbox('Yıl Seçin:', years)

    # Seçilen yıl için veri
    selected_data = monthly_sales[monthly_sales['Year'] == selected_year]

    # Grafik: Aylık satış trendi + Ortalama satış çizgisi
    fig = px.line(selected_data, x='Month_Name', y='Sales', title=f'{selected_year} Aylık Satış Trendleri', markers=True)
    fig.update_traces(line_color='royalblue')
    fig.add_hline(y=selected_data['Sales'].mean(), line_dash="dot",
                  annotation_text="Ortalama Satış", annotation_position="bottom right", line_color="red")
    fig.update_layout(xaxis_title='Ay', yaxis_title='Satış Miktarı')
    st.plotly_chart(fig)

    st.markdown("---")

    # Kümülatif Satışlar
    selected_data = selected_data.sort_values('Month')
    selected_data['Cumulative Sales'] = selected_data['Sales'].cumsum()

    fig_cumulative = px.area(selected_data, x='Month_Name', y='Cumulative Sales',
                              title=f'{selected_year} Kümülatif Satışlar', markers=True,
                              labels={'Month_Name': 'Ay', 'Cumulative Sales': 'Kümülatif Satış'})
    fig_cumulative.update_traces(line_color='green')
    st.plotly_chart(fig_cumulative)

    selected_data['Sales_Change_Percent'] = selected_data['Sales'].pct_change() * 100

    fig_change = px.bar(selected_data, x='Month_Name', y='Sales_Change_Percent',
                    title=f'{selected_year} Aylık Satış Değişim Yüzdesi', color='Sales_Change_Percent',
                    color_continuous_scale='RdYlGn')
    fig_change.update_layout(yaxis_title='Değişim (%)')
    st.plotly_chart(fig_change)


# İndirim vs Kar İlişkisi
elif options == 'İndirim vs Kar İlişkisi':
    st.header('💸 İndirim ve Kar İlişkisi')

    # İndirim oranı slider'ı
    discount_rate = st.slider('İndirim Oranı Seçin (%0 - %50)', 0, 50, 10)

    # Tolerans belirleme
    tolerance = 0.05
    filtered_data = df[(df['Discount'] >= (discount_rate / 100) - tolerance) & 
                       (df['Discount'] <= (discount_rate / 100) + tolerance)]

    # Eğer veri yoksa
    if filtered_data.empty:
        st.warning("Seçilen indirim oranına uygun veri bulunamadı.")
    else:
        # Ortalama satış ve kar
        avg_sales = filtered_data['Sales'].mean()
        avg_profit = filtered_data['Profit'].mean()

        st.subheader(f"📊 %{discount_rate} İndirim Oranı İçin Ortalama Değerler")
        st.metric(label="İndirimli Satış Ortalaması", value=f"${avg_sales:.2f}")
        st.metric(label="İndirimli Kar Ortalaması", value=f"${avg_profit:.2f}")

        # Grafik
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(data=filtered_data, x='Discount', y='Profit', color='orange', s=70)
        sns.regplot(data=filtered_data, x='Discount', y='Profit', scatter=False, color='blue', ax=ax)

        ax.set_title(f'İndirim ve Kar İlişkisi (%{discount_rate} İndirim Aralığı)', fontsize=14)
        ax.set_xlabel('İndirim (%)', fontsize=12)
        ax.set_ylabel('Kar ($)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        st.pyplot(fig)
        # Eğer 'Month' sütunu yoksa oluştur
        if 'Month' not in df.columns:
            df['Month'] = df['Order Date'].dt.month

        # Pivot tablo
        pivot = df.pivot_table(index='Category', columns='Month', values='Profit', aggfunc='sum')

        # Isı haritası
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(pivot, cmap='RdYlGn', center=0, annot=True, fmt='.0f')
        ax.set_title('Kategori ve Aylara Göre Kâr Isı Haritası')
        st.pyplot(fig)

        # Yıllık Kategori Bazlı Kar
        df['Year'] = df['Order Date'].dt.year

        category_profit = df.groupby(['Year', 'Category'])['Profit'].sum().reset_index()

        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.barplot(data=category_profit, x='Year', y='Profit', hue='Category', palette='Set2')
        ax2.set_title('Yıllara Göre Kategori Bazlı Toplam Kâr')
        st.pyplot(fig2)


# Mevsimlere Göre Satışlar
elif options == 'Mevsimlere Göre Satışlar':
    st.header('🌸 Mevsimsel Satış Dağılımı')

    # Mevsim hesaplama
    df['Month'] = df['Order Date'].dt.month
    df['Season'] = df['Month'].apply(lambda x: 'Spring' if x in [3, 4, 5] else
                                     ('Summer' if x in [6, 7, 8] else
                                      ('Fall' if x in [9, 10, 11] else 'Winter')))

    # Mevsime göre satışlar
    season_sales = df.groupby('Season')['Sales'].sum()

    fig6, ax6 = plt.subplots(figsize=(8,5))
    sns.barplot(x=season_sales.index, y=season_sales.values, palette='coolwarm', ax=ax6)
    ax6.set_title('Mevsimlere Göre Toplam Satış')
    st.pyplot(fig6)

    # Satışlar ve Kar ilişkisini görselleştiren scatter plot
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df, x='Sales', y='Profit', color='green')
    ax.set_title('Satışlar ve Kar İlişkisi')
    ax.set_xlabel('Satış ($)')
    ax.set_ylabel('Kar ($)')
    st.pyplot(fig)

    # Satış ve karların karşılaştırılması
    category_profit = df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

    fig, ax = plt.subplots(figsize=(10,6))
    category_profit.set_index('Category')[['Sales', 'Profit']].plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Kategoriye Göre Satış ve Kar Dağılımı')
    ax.set_xlabel('Kategori')
    ax.set_ylabel('Miktar ($)')
    st.pyplot(fig)

    # Order Date'ı datetime formatına çevir
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    # Yıl sütununu oluştur
    if 'Year' not in df.columns:
        df['Year'] = df['Order Date'].dt.year

    # Yıl ve kategori bazında kar analizi
    yearly_profit = df.groupby(['Year', 'Category'])['Profit'].sum().unstack().reset_index()

    fig, ax = plt.subplots(figsize=(10,6))
    yearly_profit.set_index('Year').plot(kind='area', stacked=True, ax=ax, colormap='Set2')
    ax.set_title('Yıl Bazında Kar Dağılımı')
    ax.set_xlabel('Yıl')
    ax.set_ylabel('Kar ($)')
    st.pyplot(fig)

elif options == 'ML ile Satış Tahmini':
    st.header('🤖 Makine Öğrenmesi ile Satış Tahmini')

    st.write("Aşağıdaki bilgileri doldurarak tahmini satış miktarını öğrenebilirsiniz:")

    quantity = st.number_input('Quantity (Satılan Adet)', min_value=1, max_value=10, value=3)
    discount = st.slider('Discount (İndirim Oranı)', 0.0, 0.3, 0.1)
    profit = st.number_input('Profit (Kar)', min_value=-50.0, max_value=500.0, value=50.0)

    category = st.selectbox('Category', df['Category'].unique())
    sub_category = st.selectbox('Sub-Category', df['Sub-Category'].unique())
    region = st.selectbox('Region', df['Region'].unique())
    ship_mode = st.selectbox('Ship Mode', df['Ship Mode'].unique())
    segment = st.selectbox('Segment', df['Segment'].unique())

    input_df = pd.DataFrame({
        'category': [category],
        'sub-category': [sub_category],
        'segment': [segment],
        'discount': [discount],
        'quantity': [quantity],
        'profit': [profit],
        'region': [region],
        'ship_mode': [ship_mode]
    })

    st.write("### 📄 Girdi Verileriniz:")
    st.dataframe(input_df)

    if st.button('Tahmin Yap'):
        try:
            # Burada log dönüşümü falan yapmıyoruz, çünkü model pipeline içinde hepsini hallediyor
            raw_prediction = model.predict(input_df)

            # Tahmini log dönüşümden geri alıyoruz
            prediction = float(np.expm1(raw_prediction[0]))

            if prediction < 0:
                st.error("⚠️ Tahmin edilen satış negatif çıktı, lütfen giriş değerlerini kontrol edin.")
            elif prediction > 1_000_000:
                st.warning(f"⚠️ Tahmin edilen satış çok yüksek: {prediction:,.2f} ₺")
            else:
                st.success(f"🎯 Tahmin Edilen Satış Miktarı: **{prediction:,.2f} ₺**")
                st.balloons()

        except Exception as e:
            st.error(f"⚠️ Bir hata oluştu: {str(e)}")

    st.info("Not: Girdi formatı, modelin eğitimde kullandığı formatla uyumludur.")
