# 3. TCMB EVDS API anahtarı

1. https://evds3.tcmb.gov.tr adresinde ücretsiz hesap açın.
2. Giriş yapın → sağ üstten profil → **API Anahtarı** → kopyalayın.
3. Repo kökünde `.env` dosyasına `EVDS_API_KEY=...` satırını ekleyin (`.env` git'e girmez).
4. Kullanılacak seriler: TÜFE `TP.FG.J0`, USD/TRY alış `TP.DK.USD.A.YTL`; politika faizi için
   TCMB'nin 1 hafta repo tablosu (HTML) kullanılır.
