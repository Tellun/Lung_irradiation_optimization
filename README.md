# Základní popis
Skripty pro studii hodnotící algoritmy a přístupy ozáření plic

Popis:
Projekt je navržen pro kvantifikaci a porovnání dávkové distribuce mezi naskenovaným filmem a dávkou vyexportovanou z plánovacího systému (TPS). Hlavní funkce projektu zahrnují:
  
  Načítání a předzpracování dat:
    Čtení a úprava naskenovaného filmu (soubor .tif) s využitím knihovny OpenCV. 
    Načítání CT dat a informace z DICOM headeru pro získání geometrie a měřítka.
    Načítání dávkových dat z TPS (DICOM) a jejich převod na odpovídající rozlišení.
  
  Analýza a výpočet:
    Kalibrace filmových hodnot
    Výpočet registračních parametrů mezi CT a filmem.
    Výpočet gamma mapy, histogramu a DVH křivek (srovnání mezi filmovou a TPS dávkou).
  
  Vizualizace:
    Zobrazení výsledků - gamma map, gamma histogramu, cross/in-plane profilu dávek na filmu a z TPS a DVH .
  
  Projekt využívá několik tříd, které řeší oddělené části zpracování:
    Film: Zpracování a kalibrace filmových dat.
    CT: Načítání a analýza CT dat pro získání geometrických parametrů.
    Dose: Zpracování dávkových dat z TPS, přepočet pixelů na dávku a změna rozlišení.
    Gamma: Výpočet gamové analýzy mezi filmem a dávkou TPS.
    Visualization: Komplexní vizualizační rozhraní, které kombinuje výsledky gamového výpočtu, histogramu, profilů a DVH.


Kód se nepouští přes optparser. To byla původní idea a nakonec to v kódu zůstalu misto dedikované konfigurace. 

# Na čem se pro novou verzi pracuje:

1) Modularizace kódu
   - přehlednost kódu a usnadnění menších korekcí (nebude třeba kontrolovat vliv na celý kód)
3) Přepis co největšího množství funkcí přes knihovny
   - optimalizace a robustnost
5) Sepsat unit testy
   - verifikace funkcí, robustnost
7) Dokumentace
   - přehlednost, lepší sdíletelnost
9) Konfigurační soubor nahrazující optparser
    - robustnost, přehlednost
11) Identifikace "choke" pointů a kontrola odpovídajících parametrů vstupních dat
    - robustnost 
(7) Geometrická gama?)

