<!-- TÜBİTAK 2209-A Sonuç Raporu taslağı (Türkçe teslim metni; depo dili İngilizce, bu dosya istisnadır).
     {{...}} ile işaretli yerler jüri koşusu bitince sonuç dosyalarından doldurulur; [YÜRÜTÜCÜ: ...] yerlerini
     proje yürütücüsü doldurur. Son hali resmi .docx şablonuna aktarılır. -->

# 2209/A ÜNİVERSİTE ÖĞRENCİLERİ ARAŞTIRMA PROJELERİ DESTEK PROGRAMI — SONUÇ RAPORU

**PROJE BAŞLIĞI:** **KURAL VE LLM TABANLI HİBRİT ÇOKLU AJAN SİSTEMİ İLE HİSSE ÖNERİ SİSTEMİ**

| | |
|---|---|
| Proje yürütücüsünün adı | Vedat Ali Tatlı (Erciyes Üniversitesi, Bilgisayar Mühendisliği) |
| Danışmanın adı | Arş. Gör. Hayri Öztürk |
| Projenin konusu | Borsa İstanbul (BIST-100) hisseleri için kural tabanlı (deterministik) analiz ile yerel büyük dil modeli (LLM) jürisini birleştiren, geriye dönük ve ileriye dönük olarak test edilen hibrit çoklu ajan hisse öneri sistemi |
| Başvuru numarası | 1919B012550729 |
| Proje başlangıç ve bitiş tarihleri | 01.12.2025 – 22.10.2026 |
| Proje ekibi | Vedat Ali Tatlı (yürütücü), Gizem Coşkun, Rabia Yeşil |

## 1. Giriş

Hisse senedi seçimi, sayısal göstergelerin (kârlılık, değerleme, momentum, risk) yanında bu göstergelerin
yorumlanmasını da gerektiren bir karar problemidir. Literatürdeki sistemler ya yalnızca deterministik
modellere ya da tek bir LLM çekirdeğine dayanmaktadır. Bu projede iki yaklaşım birleştirilmiştir: kural tabanlı
ajanlar her ay BIST-100 evreninden aday hisseleri üretip sıralamakta, farklı yatırım felsefelerini temsil eden
üç LLM jüri ajanı bu adayları gerekçeli puanlarla değerlendirmekte, jüri puanlarının konsensüsü nihai Top-3
öneriyi belirlemektedir.

Projenin iki hipotezi test edilmiştir:

* **H1:** LLM jürisi eklenmiş hibrit sistem, yalnızca kural tabanlı sisteme göre daha yüksek risk-ayarlı getiri
  sağlar (aynı tarih, aynı işlem maliyeti, aynı karşılaştırma ölçütü ile).
* **H2:** Jüri katmanı, kararların açıklanabilirliğini ve tutarlılığını artırır (gerekçelerin girdiye dayanması,
  tekrar edilebilirlik, girdi gürültüsüne ve istem değişikliğine dayanıklılık).

Türkiye'deki yüksek enflasyon nominal TL getirilerini tek başına anlamsız kıldığı için sonuçlar TL'nin yanında
ABD doları ve TÜFE ile reelleştirilmiş olarak, ayrıca karşılaştırma ölçütüne göre fazla getiri (aktif getiri)
olarak raporlanmıştır.

## 2. Proje konusu hakkında kısa bilgi

Sistem altı ajandan oluşur ve öneride tanımlanan beş iş paketine (İP) karşılık gelir:

| Ajan | Tür | Görev |
|---|---|---|
| Aday Havuzu (İP1) | deterministik | Her karar tarihinde o gün geçerli resmi BIST-100 üyelerini verir (hayatta kalma yanlılığı yok; endeksten çıkan ve işlem kodu değişen şirketler dahil). |
| Keşifçi (İP2) | deterministik | Fiyat, finansal tablo ve makro verileri toplar; her değeri yalnızca kamuya açıklandığı tarihten sonra kullanan (point-in-time) özellik deposunu üretir. |
| Üretici-Filtreleyici (İP3) | deterministik | Likidite, işlem süresi ve özsermaye filtrelerini uygular; Temel Güç, Değerleme ve Momentum-Risk bileşik skorlarını sektör içi sağlam z-skorlarla hesaplar; Top-10 adayı seçer. |
| Kalite/Büyüme jürisi (İP4) | LLM | Adayı kârlılık ve büyüme açısından 0–1 arası puanlar, gerekçesini yazar. |
| Değer jürisi (İP4) | LLM | Adayı değerleme açısından puanlar. |
| Risk/Zamanlama jürisi (İP4) | LLM | Adayı oynaklık, düşüş ve momentum açısından puanlar. |

Jüri skoru J, geçerli ajan puanlarının ortalamasıdır; aylık Top-3 öneri J sırasına göre seçilir (eşitlikte
bileşik skor sırası), J ≥ 0,75 eşiği ise "AL" sinyali olarak ikincil analizde kullanılır. Sistem 2018–2024
döneminde (84 aylık dönem) aylık yeniden dengelemeli, adım adım ileri (walk-forward) bir geriye dönük testle
(İP5) doğrulanmış; ayrıca gerçek zamanlı ileriye dönük "kâğıt üzerinde" bir koşu ve e-posta raporu prototipi
kurulmuştur.

Ajanlar arasındaki veri akışı ve doğrulama adımları Şekil 1'de özetlenmiştir.

![Şekil 1. Sistem mimarisi: deterministik ajanlar (İP1–İP3), maskeli girdiyle çalışan LLM jürisi (İP4), geriye dönük test, denetimler ve ileriye dönük koşu.](figures/sekil_mimari.png)

Tüm yazılım Python 3.12 ile geliştirilmiştir. Bütçe gerektirmeyen, tekrarlanabilir bir kurulum için LLM jürisi
yerel olarak (Ollama üzerinde, sabitlenmiş model sürümüyle) çalıştırılmış, tüm veriler ücretsiz ve resmi
kaynaklardan sağlanmıştır. Kaynak kod, testler ve belgeler açık kaynak olarak yayımlanmıştır:
https://github.com/vedattatli/bistmas-2209a

## 3. Rapor dönemlerinde yapılan çalışmalar

### 3.1 İP1 — Dinamik BIST-100 evreni (Aday Havuzu ajanı)
* Borsa İstanbul'un dönemsel endeks değişikliği duyuruları, resmi ağırlık dosyaları ve Borsa İstanbul
  DataStore'un ücretsiz "2000 yılından itibaren endekslerde bulunan şirketler" dosyaları kullanılarak 2017–2026
  dönemi için BIST-100/50/30 üyelikleri gün bazında yeniden kurulmuştur (2017-07 … 2026-10 arasında BIST-100'e
  girip çıkan 255 işlem kodu).
* Dönem içi olaylar (hızlı giriş, birleşme, kottan çıkış) KAP bildirimleriyle tarihlendirilmiştir. Kurulan
  üyelik, DataStore'un 135 resmi anlık görüntüsünün (BIST 100/50/30, 2017-07 … 2026-07) tamamını birebir
  üretmektedir.
* İşlem kodu değişen şirketler (ör. KOZAL→TRALT, KOZAA→TRMET, IPEKE→TRENJ, KERVT→BESLR, EFORC→EFOR) eski ve yeni
  kodlarıyla eşleştirilmiş; böylece endeksten çıkan ve adı değişen hisseler veri setinden düşmemiştir.

### 3.2 İP2 — Veri toplama ve özellik deposu (Keşifçi ajanı)
* **Fiyatlar:** İş Yatırım'ın ham ve düzeltilmiş günlük serileri; 249 hisse, 2017-01-02 … 2026-09-23, 2.436 seans,
  490.690 satır. Sermaye artırımı, bedelsiz ve temettü gibi 661 kurumsal olay tespit edilip denetlenmiş, satıcı
  verisindeki düzeltme hataları bir düzeltme tablosuyla giderilmiştir. İşlem görmeyen seanslar ayrıca ele
  alınmıştır.
* **Finansal tablolar:** KAP'ta yayımlanan ilk (sonradan düzeltilmemiş) finansal raporlar şirket bazında indirilmiş
  ve ayrıştırılmıştır: 243 şirket, 6.521 rapor, 3,26 milyon tablo satırı. Sanayi, banka (katılım bankaları
  dahil), sigorta ve diğer mali kuruluş şablonları ayrı kavram eşlemeleriyle işlenmiştir. KAP'ta raporları
  bulunmayan, endeksten erken ayrılmış birkaç şirket için İş Yatırım'ın ilk açıklanan değerleri yedek kaynak
  olarak kullanılmıştır. Son on iki ay (TTM)
  değerleri aynı rapordaki karşılaştırmalı rakamlardan hesaplanmış, TMS-29 enflasyon muhasebesine geçişte farklı
  muhasebe tabanları karıştırılmamıştır.
* **Makro veriler:** TCMB EVDS'den TÜFE (2003=100 ve 2025=100 serileri birleştirilerek; her ayın endeksi
  açıklandığı günden itibaren kullanılır) ve USD/TRY (EVDS öncelikli, eksik günlerde satıcı verisi), TCMB politika
  faizi (yürürlük tarihine göre).
* **Özellik deposu:** 84 karar tarihi × 100 endeks üyesi = 8.400 satır; her satırın dayandığı son finansal tablonun
  dönem sonu ve açıklanma tarihi ile fiyat tarihi kayıtlıdır; karar tarihinden sonra açıklanan hiçbir bilginin
  kullanılmadığı her satırda ve otomatik testlerle doğrulanmaktadır. Satırların %84,6'sı filtrelerden geçmektedir.

### 3.3 İP3 — Deterministik filtreleme ve skorlama (Üretici-Filtreleyici ajanı)
* Filtreler: en az 250 seans (yaklaşık 12 ay) işlem görme; 6 aylık ortalama günlük işlem hacminin, 5 milyon TL ile o
  ayki endeks üyelerinin %10'luk diliminden büyük olanının üzerinde olması; fiyat ve en fazla 400 günlük bir finansal
  tablo bulunması; negatif özsermayenin dışlanması.
* Göstergeler (sanayi şablonu): Temel Güç — ROIC, brüt ve faaliyet marjı, gelir ve kâr büyümesi; Değerleme —
  kazanç getirisi, defter/piyasa değeri, FAVÖK/firma değeri, serbest nakit akışı/firma değeri; Momentum-Risk —
  12-1 ay momentum, 6 aylık oynaklık, 6 aylık en büyük düşüş, 6 aylık ortalama işlem hacmi. Bankaların Temel Güç
  sütununda ROE, net faiz marjı, maliyet/gelir, özsermaye/aktif ve kâr büyümesi; sigorta ve diğer mali
  kuruluşlarda ROE, özsermaye/aktif ve kâr büyümesi; tüm mali şirketlerin Değerleme sütununda kazanç getirisi ve
  defter/piyasa değeri kullanılır.
* Normalleştirme: her ay sektör grubu içinde medyan/MAD ile sağlam z-skor (±3 kırpma); gözlemi az olan gruplar
  bir üst gruba (mali / mali olmayan) birleştirilmiştir. Üç sütun eşit ağırlıkla bileşik skoru oluşturur ve her ay
  Top-10 aday jüriye gönderilir.

### 3.4 İP4 — LLM jüri ajanları ve konsensüs
* Model: Qwen3.5 9B, yerel Ollama sunucusunda, sabitlenmiş model özeti (digest) ile; sıcaklık 0, sabit tohum,
  düşünme kipi kapalı. Ücretli API kullanılmamış, her çağrı önbelleğe alınarak koşular tekrarlanabilir ve
  kaldığı yerden sürdürülebilir hale getirilmiştir.
* Bilgi bölümlemesi: her jüri ajanı yalnızca kendi alanının göstergelerini (z-skor ve akran grubu içi yüzdelik)
  görür. Modelin geçmiş piyasa bilgisini kullanmasını (ezber/sızıntı) önlemek için istemlerde şirket adı, işlem
  kodu, tarih ve mutlak tutar bulunmaz; şirketler her ay değişen takma adlarla sunulur ve bu kurallar her istem
  için otomatik olarak doğrulanır.
* Çıktı şeması: önce gerekçe, sonra 0–1 puan; gerekçede yalnızca istemde bulunan alanlara atıf yapılabilir.
  Şemaya uymayan yanıt bir kez yeniden istenir, yine uymazsa çekimser sayılır; konsensüs en az iki geçerli oy
  ister.
* Bir tam koşu: 84 ay × 10 aday × 3 ajan = 2.520 model çağrısı (çağrı başına yaklaşık 23 saniye).

### 3.5 İP5 — Geriye dönük test ve doğrulama
* Motor: karar her ayın son seansında verilir, işlemler bir sonraki seansın ağırlıklı ortalama fiyatından yapılır;
  alım ve satımda %0,1 komisyon + 20 baz puan kayma maliyeti; işlem görmeyen hisse için nakit tutulur.
* Karşılaştırma ölçütü: o tarihteki tüm BIST-100 üyelerinin eşit ağırlıklı portföyü (aynı tarih, aynı maliyet);
  ikincil olarak yalnızca filtreden geçen hisselerin eşit ağırlıklı portföyü ve BIST-100 fiyat endeksi.
* Ölçütler: yıllık bileşik getiri (CAGR), Sharpe ve Sortino (TCMB politika faizi üzerinden), en büyük düşüş;
  aylık aktif getiri için Newey-West t-testi, blok bootstrap güven aralığı, deneme sayısına göre düzeltilmiş
  (deflated) Sharpe; 500 tohumlu rastgele jüri sıfır dağılımı; maliyet ve kottan çıkış duyarlılıkları.
* **Ön kayıt (pre-registration):** Bütün parametreler, hipotez testleri ve veri sürümleri jüri koşusundan önce
  dondurulmuş ve git etiketiyle (`prereg-v1`) kayıt altına alınmıştır. Jüri koşusu sürerken, hibrit sonuçlar
  henüz hesaplanmadan, sistemin bağımsız bir doğruluk incelemesi yapılmış (her bulgu iki ayrı doğrulayıcıyla
  sınanmıştır). Bulunan veri ve hesaplama hataları düzeltilmiş, düzeltilmiş şartname yeni jüri koşusundan önce
  `prereg-v2` olarak dondurulmuştur. Birincil sonuçlar `prereg-v2`'ye, ilk kayıt (`prereg-v1`) ise ikincil
  analiz olarak raporlanmıştır.

### 3.6 İleriye dönük koşu ve e-posta raporu prototipi
Her ayın son seansından sonra sistemin güncel verilerle Top-3 kararını üretip mühürleyen (zaman damgalı karar
dosyası ve özet sertifikası) ve sonucu e-posta raporu olarak gönderen bir modül geliştirilmiştir. Karar tarihi
yalnızca tamamlanmış aylar için seçilir, resmi tatiller dikkate alınır; planlanan uygulama seansı ayın ilk
seansından sonraya kalan kararlar "geç" olarak etiketlenir. Örnek olarak 30 Eylül 2026 verileriyle üretilen ve 21 Ekim 2026 seansında uygulanmak
üzere mühürlenen karar (gecikmeli üretildiği için "geç" etiketli; özet değeri `d396e05a…`) şöyledir: kural tabanlı
Top-10 içinden jüri TUPRS (J = 0,883), KTLEV (0,867) ve THYAO (0,850) hisselerini seçmiş, üç ajanın tamamı geçerli
oy vermiştir. Raporda her hisse için üç ajanın gerekçesi yer alır (ör. değer ajanı THYAO için "kazanç getirisi,
defter/piyasa ve FAVÖK getirisi akran grubunda en üst yüzdelikte; serbest nakit akışı getirisi biraz geride"
gerekçesiyle 0,95 puan vermiştir). Rapor metninin tamamı Ek-1'dedir; e-posta gönderimi aynı içeriği HTML olarak
iletir. Bu çıktılar yatırım tavsiyesi değildir.

### 3.7 Yazılım altyapısı ve iş bölümü
* Açık kaynak depo (`bistmas`, MIT lisansı): yaklaşık 15.900 satır Python, yaklaşık 390 otomatik test, modüller arası
  bağımlılık kuralları, her deneyin veri sürümü ve yapılandırma özetiyle kaydedildiği bir deney defteri.
* İş bölümü: Vedat Ali Tatlı — veri katmanı (İP1, İP2) ve proje yönetimi; Gizem Coşkun — skorlama ve geriye dönük
  test (İP3, İP5); Rabia Yeşil — jüri ajanları, denetimler, ileri koşu ve raporlama (İP4).

### 3.8 Üretken yapay zekâ (ÜYZ) kullanım beyanı
"TÜBİTAK Destek Süreçlerinde Üretken Yapay Zekânın Sorumlu ve Güvenilir Kullanımı Rehberi" uyarınca beyan edilir:
* **Araç:** Anthropic Claude (Claude Code kodlama asistanı; Claude Max aboneliği, bkz. Bölüm 6).
* **Kullanıldığı aşamalar:** yazılımın kodlanması ve testlerin yazılması, veri işleme ve analiz betikleri, kod
  incelemesi, proje belgelerinin ve bu raporun metin taslağı.
* **Kullanımın niteliği ve kapsamı:** Sistem tasarımı, iş paketleri, veri kaynakları, yöntem ve parametre
  kararları proje ekibi tarafından verilmiş; kod ve metin taslakları bu kararlar doğrultusunda araçla üretilmiş,
  ekip tarafından incelenip düzenlenmiştir. Kodun doğruluğu otomatik testlerle, sonuçlar veri dosyalarıyla
  karşılaştırılarak doğrulanmıştır; ön kayıt ve bağımsız doğruluk incelemesi bu doğrulamanın parçasıdır. Rapordaki
  tüm sayılar sistemin ürettiği sonuç dosyalarından alınmıştır. Metnin ve sonuçların sorumluluğu proje ekibine
  aittir.
* Sistemin kendisinde araştırma konusu olarak kullanılan LLM jürisi (Qwen3.5 9B) yerel olarak çalıştırılmış olup bu
  beyanın kapsamı dışındadır.


### 3.9 Çalışma takvimi ve başarı ölçütlerinin gerçekleşme durumu
Öneride her iş paketi için bir dönem ve başarı ölçütü tanımlanmıştı. Aşağıdaki tablo bu ölçütleri ve
gerçekleşme durumunu karşılaştırmaktadır.

| İş paketi (önerideki dönem) | Başarı ölçütü (öneri) | Gerçekleşme |
|---|---|---|
| İP-1 Aday havuzu ajanı (12/2025) | BIST-100 evreni; eksik veri oranı < %2 | Gerçekleşti. Zaman noktası BIST-100 evreni kuruldu ve Borsa İstanbul'un 135 resmi bileşen listesiyle birebir doğrulandı. Her karar tarihinde endeks üyelerinin %99,3–100'ünün işlem fiyatı mevcuttur. |
| İP-2 Veri toplama ve özellik deposu (01–02/2026) | ETL akışı otomatik; özellik deposundan sorgu ≤ 300 ms | Gerçekleşti, kaynak değişikliğiyle. Finnhub yerine ücretsiz kamu kaynakları (KAP, İş Yatırım, Borsa İstanbul DataStore, TCMB EVDS) kullanıldı. ETL, kaldığı yerden devam edebilen komut satırı işleri olarak otomatiktir. Bir karar tarihinin özellik deposu sorgusu yaklaşık 2 ms sürmektedir; finansal tablo kapsamı %99,7'dir. |
| İP-3 Faktör hesaplama ve normalizasyon (03–04/2026) | Tüm faktörler hesaplanmış; faktör korelasyon raporu | Gerçekleşti, sadeleştirilerek. Dört faktör grubu yerine üç sütun (temel güç, değerleme, momentum-risk) sektör içi sağlam z-skorla hesaplandı. Sütunlar arası Spearman korelasyonu 0,17–0,28'dir; sütunlar birbirinden büyük ölçüde bağımsızdır. |
| İP-3 Filtreleme ve ön eleme (05/2026) | Filtre sonrası eksik veri oranı < %1; Top-10 listesi | Gerçekleşti. 84 ayın tamamında Top-10 listesi üretildi; Top-10 adayların tamamında üç sütun da hesaplıdır. |
| İP-4 Konsensüs ve raporlama modülü (05–08/2026) | Top-3 otomatik üretiliyor; özet rapor şablonu (e-posta ve ayrıntı) | Gerçekleşti. Üç LLM jüri ajanı, konsensüs ve Top-3 seçimi; aylık karar raporu ve e-posta raporu prototipi geliştirildi (Ek-1). |
| İP-5 Sistem doğrulaması (08–10/2026) | CAGR, Sharpe, Sortino, en büyük düşüş; kıyaslama | Gerçekleşti. 84 aylık geriye dönük test; TL, USD ve reel metrikler; eşit ağırlıklı BIST-100 ile kıyaslama ve istatistiksel testler (4.1–4.2). |
| Sonuç analizi ve raporlama (11/2026) | Nihai rapor; CAGR ≥ %15, Sharpe > 1,2; açık kaynak depo; Llama 3 70B geçiş planı | Kısmen. Rapor ve açık kaynak depo tamamlandı. Nominal CAGR hedefi tüm portföylerde aşıldı, ancak bu enflasyondan kaynaklanmaktadır; politika faizi düşülerek hesaplanan Sharpe > 1,2 hedefine ulaşılamadı. 70B model geçişi donanım kısıtı nedeniyle yapılmadı; yerine farklı model aileleriyle çapraz jüri sonraki çalışma olarak önerildi (4.4). |

## 4. Sonuç

### 4.1 Kural tabanlı sistem (2018–2024, 84 ay)
| Portföy | CAGR (TL) | Sharpe | CAGR (USD) | CAGR (reel) | Aylık aktif getiri (t) |
|---|---|---|---|---|---|
| Kural tabanlı Top-3 | %40,8 | 0,56 | %2,3 | %4,2 | −%0,05 (−0,07) |
| Kural tabanlı Top-5 | %36,3 | 0,48 | −%1,0 | %0,9 | −%0,37 (−0,60) |
| Kural tabanlı Top-10 | %43,7 | 0,67 | %4,4 | %6,4 | +%0,06 (0,12) |
| BIST-100 eşit ağırlık (ölçüt) | %44,1 | 0,73 | %4,7 | %6,7 | — |
| Filtreden geçenler eşit ağırlık | %42,3 | 0,68 | %3,4 | %5,3 | — |

Kural tabanlı skor tek başına eşit ağırlıklı BIST-100 portföyünü anlamlı biçimde geçememiştir. Nominal TL
getiriler önerideki %15 CAGR ölçütünün çok üzerindedir; ancak bu, dönemin yüksek enflasyonunu yansıtır ve
karşılaştırma ölçütü de aynı düzeydedir. Reel ve dolar bazında getiriler tek haneli kalmıştır. Önerideki
Sharpe > 1,2 ölçütüne, risksiz getiri olarak TCMB politika faizi düşüldüğünde, karşılaştırma ölçütü dahil hiçbir
portföy ulaşamamıştır (yüksek faiz döneminde hisse getirilerinin risksiz getiriyi aşan kısmı sınırlı kalmıştır).

### 4.2 Hibrit sistem — H1
Birincil analizde (`prereg-v2`) jüri 84 ayın tamamında 2.520 değerlendirmenin hepsini geçerli biçimde tamamlamış
(çekimser oranı %0), her aday için J hesaplanmıştır.

| Portföy | CAGR (TL) | Sharpe | En büyük düşüş | CAGR (USD) | CAGR (reel) | Aylık aktif getiri (NW t) | Hibrit − kural (NW t, p) |
|---|---|---|---|---|---|---|---|
| Hibrit Top-3 | %35,0 | 0,45 | −%41,5 | −%1,9 | %0,0 | −%0,48 (−0,65) | −%0,43 (−0,56; 0,58) |
| Kural tabanlı Top-3 | %40,8 | 0,56 | −%40,8 | %2,3 | %4,2 | −%0,05 (−0,07) | — |
| Hibrit Top-5 | %39,0 | 0,54 | −%37,8 | %1,0 | %2,9 | −%0,22 (−0,29) | +%0,15 (0,32; 0,75) |
| Kural tabanlı Top-5 | %36,3 | 0,48 | −%41,2 | −%1,0 | %0,9 | −%0,37 (−0,60) | — |
| Hibrit = kural tabanlı Top-10 | %43,7 | 0,67 | −%40,3 | %4,4 | %6,4 | +%0,06 (0,12) | 0 |
| BIST-100 eşit ağırlık (ölçüt) | %44,1 | 0,73 | −%35,5 | %4,7 | %6,7 | — | — |

![Şekil 2. Hibrit, kural tabanlı ve karşılaştırma portföylerinin 2018–2024 net değer eğrileri (sol: TL, sağ: ABD doları; logaritmik ölçek, işlem maliyetleri düşülmüş).](figures/sekil1_nav.png)

* **H1 desteklenmemiştir.** Hibrit Top-3, kural tabanlı Top-3'e göre ayda ortalama %0,43 daha düşük getiri
  sağlamıştır; fark istatistiksel olarak anlamlı değildir (Newey-West t = −0,56, p = 0,58; aktif getirinin %95
  bootstrap güven aralığı [−%1,68; +%0,86]). Top-5'te hibrit sistem ayda %0,15 öndedir, bu fark da anlamlı
  değildir. Jüriye her ay 10 aday gittiği için Top-10 iki sistemde aynı portföydür.
* Aynı adaylara rastgele puan veren 500 "rastgele jüri" ile karşılaştırıldığında hibrit Top-3'ün getirisi bu
  dağılımın 41. yüzdeliğindedir (rastgele jürilerin CAGR ortalaması %36,6; %5–%95 aralığı %24,4–%49,1): jüri
  sıralaması, aday listesinin rastgele sıralanmasından ayırt edilememektedir.
* Jüri skoru J ile bileşik skor arasındaki aylık Spearman korelasyonu ortalama 0,22'dir; J, bileşik skoru tekrar
  etmemekte ancak ertesi ay getirisi için bileşik sıranın ötesinde ek bilgi de taşımamaktadır (J'nin kesitsel
  regresyon katsayısı 0,005; t = 0,10).
* "AL" sinyali varyantı (J ≥ θ olan adaylar, yoksa nakit): θ = 0,75'te ayda ortalama 4,3 AL sinyali, yalnızca 2 ay
  nakit, TL CAGR %39,3 (aylık aktif −%0,23, t = −0,30); θ = 0,70 ve 0,80'de sonuç değişmemektedir (aktif t −0,03 ve
  −1,06).
* Hiçbir portföyün aktif getirisi Bonferroni düzeltmesi öncesinde de sonrasında da (8 yapılandırma, α = 0,0063)
  anlamlı değildir; deneme sayısına göre düzeltilmiş Sharpe olasılığı hibrit Top-3 için 0,02'dir.

### 4.3 Açıklanabilirlik ve tutarlılık — H2
![Şekil 3. Sol: jüri ajanlarının puan dağılımı (her ajan 840 değerlendirme); sağ: aylık jüri skoru J ile kural tabanlı bileşik sıra.](figures/sekil2_juri.png)

* **Gerekçelerin girdiye dayanması.** 2.520 gerekçenin %99,7'si yalnızca istemde verilen göstergelere atıf yapmıştır
  (ajanlara göre %100, %99,9 ve %99,3). Gerekçelerde şirket adı, işlem kodu, yıl veya ay adı gibi bilgi sızıntısı
  bulunmamıştır (sızıntı bayrağı oranı %0).
* **Tekrarlanabilirlik.** Örneklemdeki 17 aday, sıcaklık 0,7 ile beşer kez yeniden puanlanmıştır. Sınıf içi
  korelasyon (ICC) Kalite/Büyüme ajanında 0,84, Risk/Zamanlama ajanında 0,85, Değer ajanında 0,55'tir. İlk iki
  ajanın puanları rastgeleliğe karşı tutarlı, Değer ajanınınki orta düzeyde tutarlıdır.
* **Sağlamlık.** Tabakalı olarak seçilen 6 karar ayında jüri iki değişik koşulda yeniden çalıştırılmıştır
  (360 çağrı). İlk koşulda istemdeki tüm z-skorlarına ±0,25 aralığında rastgele gürültü eklenmiştir. J sıralaması
  ile asıl sıralama arasındaki Kendall τ ortalama 0,63, Top-3 örtüşmesi ortalama %78'dir (3 hisseden yaklaşık 2,3'ü
  aynı kalır). İkinci koşulda istemler aynı içerikle yeniden yazılmıştır. Bu koşulda τ = 0,57, Top-3 örtüşmesi %78'dir.
  Jüri kararları küçük girdi ve ifade değişikliklerine karşı orta düzeyde sağlamdır. Top-3'ün yaklaşık üçte biri
  sınırdaki adaylar arasında yer değiştirmektedir; bu da J'nin ek getiri bilgisi taşımamasıyla tutarlıdır.
* **Ajanlar arası uyum.** Ajanların aynı aday için verdikleri puanlar arasındaki ortalama ikili Spearman korelasyonu
  −0,10, Krippendorff α (aralık) −0,19'dur. Her ajan farklı bir gösterge grubunu gördüğü için bu bekleniyordu:
  jüri tek bir görüşün tekrarı değil, üç ayrı bakış açısıdır. Puan dağılımları da farklıdır (Şekil 3). Değer ajanı
  en cömert, Risk/Zamanlama ajanı en eleştirel ajandır.
* **Ezber ve sızıntı denetimleri** (84 adaylık tabakalı örneklem):
  * Kimlik tahmini: maskelenmiş göstergelerden şirketi doğru tahmin oranı %0'dır (şans düzeyi ~%1). Dönem (yıl ve
    çeyrek) doğru tahmin oranı %3,6'dır (şans düzeyi %3,4). Model şirketleri göstergelerden tanıyamamaktadır.
  * Doğrudan soru: gerçek şirket adı ve tarih verilerek kapanış fiyatı sorulduğunda ortalama mutlak yüzde hata %90,
    %10 içinde doğru tahmin oranı %4,8, ertesi ay yön tahmini doğruluğu %45'tir. Model 2018–2024 fiyatlarını
    ezberlememiştir. Buna karşın aynı sorularda "bilmiyorum" yerine yön belirtme olasılığı ortalama 0,85'tir;
    yani model bilmediği konuda kendinden emin yanıt verme eğilimindedir. Bu da jüri istemlerinde kimlik ve tarihin
    gizlenmesinin gerekliliğini doğrulamaktadır.
  * Tarih kolları: aynı istemde tarih eklendiğinde, gerçek şirket ve tarih açıklandığında ya da tarih 3 yıl ileri
    veya 7 yıl geri kaydırıldığında puanlar sistematik olarak değişmemiştir. Ortalama fark 0,002'nin altında,
    ortalama mutlak fark 0,04–0,06, maskeli puanlarla Spearman korelasyonu 0,89–0,94'tür. Kimliğin açılması
    puanların yalnızca %21'ini 0,1'den fazla değiştirmiştir.
* **İnsan değerlendirmesi.** Ön kayıtta öngörülen, 30 jüri gerekçesinin üç değerlendirici tarafından kör
  puanlanması zaman kısıtı nedeniyle yapılamamış ve sonraki çalışmaya bırakılmıştır; değerlendirme formları
  hazırlanmıştır (`docs/report/rubric/`).
* **Sonuç (H2).** Jüri gerekçeleri girdiye dayalı, sızıntısız, büyük ölçüde tekrarlanabilir ve orta düzeyde sağlamdır; model ne şirket
  kimliğini ne de geçmiş fiyatları kullanabilmektedir. Bu yönüyle jüri katmanı, kural tabanlı sıralamaya her aday
  için okunabilir ve denetlenebilir bir gerekçe eklemektedir; ancak bu açıklanabilirlik getiri üstünlüğüne
  dönüşmemiştir (4.2).

### 4.4 Genel değerlendirme
* **Ana bulgu.** Bu projede kurulan hibrit sistem uçtan uca çalışmaktadır: veri toplama, hayatta kalma yanlılığı
  giderilmiş evren, zaman açısından sızıntısız özellik deposu, kural tabanlı aday üretimi, LLM jürisi, geriye dönük
  test, denetimler ve ileriye dönük karar üretimi. Ancak 2018–2024 döneminde yerel 9 milyar parametreli bir
  modelden oluşan jüri, kural tabanlı Top-10 adaylar arasından daha iyi getiri sağlayan hisseleri seçememiştir.
  H1 desteklenmemiştir. Bu "olumsuz" sonuç, ön kayıt, körleme ve rastgele jüri karşılaştırmasıyla sınandığı için
  güvenilirdir: LLM'in finansal göstergeleri tutarlı ve okunabilir biçimde yorumlaması (bkz. 4.3), tek başına
  ek getiri anlamına gelmemektedir.
* **İşlem maliyeti etkisi.** Jüri sıralaması aydan aya daha fazla değiştiği için hibrit Top-3'ün aylık portföy
  devri %48'dir (kural tabanlıda %30). Maliyetsiz senaryoda bile hibrit Top-3 (TL CAGR %39,7) kural tabanlı Top-3'ün
  (%43,9) gerisindedir; yani fark yalnızca maliyetten kaynaklanmamaktadır.
* **İstatistiksel güç.** 84 aylık örneklemde %80 güçle saptanabilecek en küçük aylık aktif getiri Top-3 için
  yaklaşık %2,1'dir (yıllık ~%25). Daha küçük etkiler bu dönemde ayırt edilemez; sonuç, "jüri büyük bir katkı
  sağlamamaktadır" biçiminde okunmalıdır.
* **Hedeflerle karşılaştırma.** Öneride hedeflenen nominal %15 CAGR tüm portföylerde aşılmış, ancak bu dönemin
  enflasyonundan kaynaklanmıştır; Sharpe > 1,2 hedefine politika faizi düşüldüğünde hiçbir portföy ulaşamamıştır.
  Projenin yazılım, açık kaynak ve raporlama hedefleri gerçekleştirilmiştir.
* **Ön kayıt v1 ile v2.** İlk dondurulan uygulama (prereg-v1) da jüriyle sonuna kadar koşturulmuştur (84 ay, 2.520 değerlendirme, çekimser oranı %0). Bu sürümde hibrit Top-3'ün TL CAGR'si %31,7, kural tabanlı Top-3'ünkü %34,2'dir. Hibrit ile kural tabanlı arasındaki aylık fark −%0,18'dir (NW t −0,24; p 0,81). Hibrit, rastgele jüri dağılımının 37. yüzdeliğinde kalmıştır. Düzeltilmiş v2'de aynı fark −%0,43'tür (t −0,56; p 0,58), rastgele jüri yüzdeliği 41'dir. İki sürümde de jüri portföy devrini artırmış (%50 ve %48; kural tabanlıda %32 ve %30) ve ek getiri sağlamamıştır. H1'in desteklenmemesi, v1'den v2'ye yapılan uygulama düzeltmelerinden bağımsızdır.
* **Sınırlılıklar.** Tek bir model ailesi (Qwen) kullanılmıştır; 2018–2024 dönemi modelin eğitim verisi kapsamında
  kaldığı için ezber/sızıntı denetimleri yapılmış (bkz. 4.3), en temiz test ise eğitim tarihinden sonraki ileriye
  dönük koşudur ve bu koşu henüz kısa bir süreyi kapsamaktadır. Jüriye yalnızca sayısal göstergeler verilmiştir.
* **Gelecek çalışmalar.** Farklı model aileleriyle (Gemma, Llama) çapraz jüri; KAP bildirim metinlerinin jüriye
  maskeleme denetimiyle eklenmesi; aday havuzunun genişletilmesi (Top-20/30) ile jürinin seçme payının artırılması;
  ileriye dönük aylık koşunun sürdürülmesi.

## 5. Çıktılar (yayınlar, sunumlar vb.)
* Açık kaynak yazılım deposu: `bistmas` (GitHub; veri toplama, skorlama, jüri, geriye dönük test, denetim ve
  raporlama modülleri, testler ve belgeler): https://github.com/vedattatli/bistmas-2209a (herkese açık). Ön kayıt
  etiketlerini içeren tam geliştirme geçmişi kişisel iletişim bilgileri barındırdığı için özel depoda tutulmakta,
  talep hâlinde paylaşılabilmektedir.
* Ön kayıt belgeleri (`prereg-v1`, `prereg-v2`), veri sözlüğü, veri yönetişimi belgesi ve deney defteri.
* İleriye dönük karar raporu ve e-posta raporu prototipi (Ek-1: 2026-09 kararı).
* Sonuç tabloları ve şekiller (TL/USD/reel net değer eğrileri, yıllık aktif getiriler, jüri skor tanılamaları).
* Yazılım deposunun README dosyasında ve bu raporda projenin TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destek Programı kapsamında desteklendiği belirtilmiştir.
* Proje süresi içinde yayımlanmış veya gönderilmiş bir bildiri ya da makale bulunmamaktadır; sonuçların bildiri
  olarak hazırlanması sonraki çalışmalara bırakılmıştır.

## 6. Proje ile ilgili harcama kalemleri hakkında ayrıntılı bilgi
Projeye 9.000 TL destek sağlanmıştır. Destek, önerideki "hizmet alımı" kalemi kapsamında yapay zekâ hizmet alımına
kullanılmıştır:

| Kalem | Açıklama | Tutar |
|---|---|---|
| Hizmet alımı — yapay zekâ | Anthropic Claude Max aboneliği, 1. ay (Google Play, sipariş no GPA.3368-3344-9472-70602) | 04.08.2026 — 9.999,88 TL (1.666,65 TL KDV dahil) |
| Hizmet alımı — yapay zekâ | Aynı aboneliğin yenilemesi (sipariş no GPA.3368-3344-9472-70602..0) | 04.09.2026 — 9.999,88 TL (1.666,65 TL KDV dahil) |
| Hizmet alımı — yapay zekâ | Aboneliğin yeniden başlatılması (sipariş no GPA.3361-7364-6447-12686) | 18.10.2026 — 6.999,99 TL (1.166,66 TL KDV dahil) |
| Toplam | | 26.999,75 TL |

Claude Max aboneliği, yazılım geliştirme, veri analizi ve raporlamada kullanılan yapay zekâ destekli kodlama ve
analiz aracıdır (bkz. 3.8). Abonelik bedelinin 9.000 TL'si TÜBİTAK desteğinden, kalan 17.999,75 TL'si yürütücü
tarafından karşılanmıştır; iade edilecek kullanılmamış tutar yoktur. Ödemeler Google Play üzerinden yapıldığı için
harcama belgeleri, yürütücü adına düzenlenmiş Google Play sipariş makbuzlarıdır (KDV gösterilmiştir) ve Ek-2'de verilmiştir.

Veri için ücret ödenmemiştir: fiyat ve finansal tablo verileri ücretsiz kamu kaynaklarından (KAP, Borsa İstanbul
DataStore ücretsiz dosyaları, İş Yatırım, TCMB EVDS) sağlanmış; araştırma konusu olan LLM jürisi ekibin kendi
bilgisayarında yerel olarak çalıştırıldığı için API ücreti doğmamıştır.

---
Proje yürütücüsü: Vedat Ali Tatlı — imza  Danışman: Hayri Öztürk — imza  Tarih: 22.10.2026
