# 1. MKK API Portal (KAP veri servisleri) — kayıt adımları

**Amaç:** KAP bildirimlerine (finansal raporlar, özel durum açıklamaları, endeks duyuruları)
resmi ve ücretsiz API üzerinden erişmek.

1. https://apiportal.mkk.com.tr adresinde **Kayıt Ol** ile hesap açın (kurumsal e-posta olarak
   `[üniversite e-postası]` kullanın; onay sürecinde akademik kimlik avantaj sağlar).
2. Hesap onaylandıktan sonra **Uygulama Oluştur** → uygulama adı `bistmas-erciyes-2209a`,
   açıklama: "TÜBİTAK 2209-A araştırma projesi: BIST hisse seçimi için hibrit çoklu ajan sistemi".
3. Oluşan **API Key**'i kaydedin (repoya yazmayın; `.env` dosyasına `MKK_API_KEY=` olarak koyun).
4. **API Ürünleri** altından KAP veri yayın servislerine (Disclosures, DisclosureDetail,
   DownloadAttachment, Members, MemberDetail, MemberSecurities) abone olun.
5. Üretim erişimi için IP beyanı istenirse ev/okul IP adresinizi bildirin; test ortamı
   (`apigwdev.mkk.com.tr`) için IP gerekmez.
6. Sorun olursa `kapdestek@mkk.com.tr` adresine aşağıdaki e-postayı gönderin.

## E-posta taslağı (kapdestek@mkk.com.tr)

Konu: MKK API Portal – akademik kullanım için KAP veri servisi erişimi

Sayın Yetkili,

Erciyes Üniversitesi Bilgisayar Mühendisliği Bölümü 4. sınıf öğrencisiyim. TÜBİTAK 2209-A
Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı kapsamında desteklenen
"Kural ve LLM Tabanlı Hibrit Çoklu Ajan Sistemi ile Hisse Öneri Sistemi" başlıklı projeyi
(Başvuru No: 1919B012550729, Danışman: Arş. Gör. Hayri Öztürk) yürütmekteyim.

Projede 2017–2024 dönemi için BIST-100 şirketlerinin KAP'ta yayımlanan finansal rapor
bildirimlerinin yayın tarih-saatlerine ve rapor içeriklerine (XBRL/HTML), ayrıca Borsa İstanbul
A.Ş. tarafından yapılan endeks dönemsel değişiklik bildirimlerine ihtiyaç duyulmaktadır. Veriler
yalnızca akademik amaçla, yeniden dağıtılmaksızın kullanılacak; sonuçlar açık kaynak kod ve
bilimsel yayın olarak paylaşılacaktır.

MKK API Portal üzerinde `bistmas-erciyes-2209a` adlı uygulamam için KAP veri yayın servislerine
(Disclosures, DisclosureDetail, DownloadAttachment, Members, MemberDetail, MemberSecurities)
erişim ve gerekli ise IP tanımlaması konusunda yardımınızı rica ederim. Tarihsel bildirimlerin
(2017 ve sonrası) servis üzerinden toplu olarak alınıp alınamayacağı konusunda da bilgi
verebilirseniz sevinirim.

Saygılarımla,
Vedat Ali Tatlı
Erciyes Üniversitesi, Bilgisayar Mühendisliği
Tel: [telefon] · E-posta: [üniversite e-postası]
