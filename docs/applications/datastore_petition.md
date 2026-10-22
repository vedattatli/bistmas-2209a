# 2. Borsa İstanbul DataStore — hesap ve akademik başvuru

**Amaç:** (a) 0 TL fiyatlı resmi dosyalar: "2000 Yılından İtibaren Endekslerde Bulunan
Şirketler (BIST 30/50/100)" (ürün tipi 3184, `exsrkYYYY.zip`), "Borsa İstanbul Pazarlarından
Sürekli Olarak Çıkarılan Şirketler", "Pay Adı ve Kodu Değişiklikleri", "İlk İşlem Verileri";
(b) akademik fiyatlandırma ile günlük bülten (fiyat) verisi.

## Adımlar
1. https://datastore.borsaistanbul.com → **Üye Ol**. Önce normal bireysel profil ile deneyin;
   ücretsiz (0 TL) dosyalar bu profille indirilebiliyorsa akademik profil gerekmez.
2. Giriş yaptıktan sonra ürün tipi 3184 sayfasından `exsrk2017.zip` … `exsrk2025.zip` dosyalarını
   kütüphaneye ekleyip indirin; `data/raw/datastore/` altına koyun (git'e girmez).
3. Aynı şekilde "Sürekli Olarak Çıkarılan Şirketler", "Pay Adı ve Kodu Değişiklikleri" ve
   "İlk İşlem Verileri" dosyalarını indirin.
4. Günlük bülten verisi (fiyatlar) için akademik profil gerekir: aşağıdaki dilekçeyi Borsa
   İstanbul'un şablonuna (https://www.borsaistanbul.com/files/datastore-da-akademik-basvuru-icin-dilekce-ornegi.doc)
   göre doldurup danışmanınıza imzalatın ve `vyk-marketing@borsaistanbul.com` adresine gönderin.

## Dilekçe metni (şablona uyarlayın)

BORSA İSTANBUL A.Ş.
Pazarlama ve Satış Bölümü'ne

Erciyes Üniversitesi Mühendislik Fakültesi Bilgisayar Mühendisliği Bölümü lisans öğrencisiyim.
TÜBİTAK 2209-A programı kapsamında desteklenen "Kural ve LLM Tabanlı Hibrit Çoklu Ajan Sistemi
ile Hisse Öneri Sistemi" başlıklı araştırma projesini (Başvuru No: 1919B012550729) danışmanım
Arş. Gör. Hayri Öztürk ile yürütmekteyim.

Projede 01.01.2017–31.12.2024 dönemi için Pay Piyasası günlük bülten verileri (kapanış, ağırlıklı
ortalama fiyat, işlem hacmi, düzeltilmiş fiyat katsayıları), BIST 30/50/100 endeks bileşen
listeleri ve endeks kapanış değerleri kullanılacaktır. Veriler yalnızca akademik amaçla
kullanılacak, üçüncü kişilerle paylaşılmayacak ve yayınlarda Borsa İstanbul kaynak gösterilecektir.

DataStore üzerinden akademik başvuru kapsamında söz konusu verilere ücretsiz ya da indirimli
erişim sağlanması hususunda gereğini arz ederim.

Ad Soyad: Vedat Ali Tatlı · T.C. Kimlik No: (dilekçede) · Tel: [telefon]
E-posta: [üniversite e-postası]
Danışman: Arş. Gör. Hayri Öztürk (hayriozturk@erciyes.edu.tr)
Tarih / İmza (öğrenci ve danışman)
