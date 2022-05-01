### 1. Check-out spullen
Ga naar: https://github.com/deltares/hydrolib
Open de groene Code tab. Clone of download de files als zip in de gewenste directory. 


### 2. Aanmaken environment
Ga naar de directory hydrolib\contrib\d2hydro\environments\ via de verkenner of total commander
Open Command Prompt (Type in Address Bar: cmd)
Type nu hetvolgende in de geopende command prompt:

```
conda env create -f environment_dev.yml
```

Na installatie kun je je environment activeren door:

```
conda activate hydrolib
```

### 3. Installeren hydrolib

Ga naar de directory hydrolib\contrib\d2hydro\modules\stowa_bui_generator. Open command prompt (type in adress bar: cmd)
In de geactiveerde environment kun je hydrolib installleren via de command prompt (vergeet niet de punt!):
```
conda activate hydrolib
```
```
pip install -e .
```

Ga naar de directory hydrolib\contrib\d2hydro\modules\case_management_tools. Open command prompt (type in adress bar: cmd)
Type nu hetvolgende in de geopende command prompt :

```
conda activate hydrolib
```
```
pip install -e .
```

### 4. Toevoegen aan environment van hydrolib-core
Open de command prompt (type in adress bar: cmd)
Type nu hetvolgende in de command prompt:

```
conda activate hydrolib
```
```
pip install git+https://github.com/Deltares/HYDROLIB-core.git&fix/224_rainfallrunoff_save
```

### 5. Downloaden van voorbeelddata
Download de data van https://drive.google.com/drive/folders/1Wl-7keuO-FGzmKbrFJn9V59oLIAtNJnG?usp=sharing en voeg de unzipped map "dellen" toe aan hydrolib\contrib\d2hydro\data
Check: Onder data heb je nu een map Dellen met hierin 3 gezipte bestanden en 1 json.
Maak nu onder data een lege map "stochast" aan

### 6. draaien van jupyter notebook
Ga naar de directory hydrolib\contrib\d2hydro\notebooks\ in environment
```
conda activate hydrolib
```
```
jupyter notebook
```
```
Selecteer totale workflow.ipynb
```
```
Click op run
```

### 7. Resultaat
Wanneer jupyter notebook klaar is, staan de resultaten in de map data/stochast. Je ziet de volgende aangemaakte folders
- boundary_conditions\
- cases\  
- initial_conditions\
- models\
- manifest.json
