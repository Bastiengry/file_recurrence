import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import glob
import os
from docx import Document
import fitz

# List of french stop words
STOP_WORDS_FR = [
    'ai', 'aie', 'aient', 'aies', 'ait', 'alors', 'as', 'au', 'aucun', 'aura', 'aurai', 'auraient', 'aurais', 'aurait', 'auras', 'aurez', 'auriez', 'aurions', 'aurons', 'auront', 'aux', 'avaient', 'avais', 'avait', 'avec', 'avez', 'aviez', 'avions', 'avons', 'ayant', 'ayante', 'ayantes', 'ayants', 'ayez', 'ayons', 'bon', 'car', 'ce', 'ceci', 'cela', 'ces', 'cet', 'cette', 'choix', 'chez', 'combien', 'comme', 'comment', 'dans', 'de', 'dedans', 'dehors', 'depuis', 'des', 'deux', 'devrait', 'doit', 'donc', 'dos', 'droite', 'du', 'début', 'elle', 'elles', 'en', 'encore', 'es', 'est', 'et', 'eu', 'eue', 'eues', 'eurent', 'eus', 'eusse', 'eussent', 'eusses', 'eussiez', 'eussions', 'eut', 'eux', 'eûmes', 'eût', 'eûtes', 'faire', 'fait', 'faites', 'fois', 'font', 'force', 'furent', 'fus', 'fusse', 'fussent', 'fusses', 'fussiez', 'fussions', 'fut', 'fûmes', 'fût', 'fûtes', 'grâce', 'haut', 'hors', 'ici', 'il', 'ils', 'je', 'juste', 'la', 'le', 'les', 'leur', 'leurs', 'lui', 'ma', 'maintenant', 'mais', 'me', 'mes', 'moi', 'moins', 'mon', 'mot', 'même', 'ne', 'ni', 'nommés', 'notre', 'nous', 'nouveaux', 'on', 'ont', 'ou', 'où', 'par', 'parce', 'parole', 'pas', 'personne', 'peu', 'peut', 'pièce', 'plupart', 'plus', 'plusieurs', 'pour', 'pourquoi', 'qu', 'quand', 'que', 'quel', 'quelle', 'quelles', 'quels', 'qui', 'sa', 'sans', 'se', 'sera', 'serai', 'seraient', 'serais', 'serait', 'seras', 'serez', 'seriez', 'serions', 'serons', 'seront', 'ses', 'si', 'sien', 'soient', 'sois', 'soit', 'sommes', 'son', 'sont', 'sous', 'soyez', 'soyons', 'suis', 'sur', 'ta', 'tandis', 'te', 'tel', 'telle', 'telles', 'tels', 'tes', 'toi', 'ton', 'tous', 'tout', 'toute', 'toutes', 'très', 'tu', 'un', 'une', 'valeur', 'vers', 'voie', 'voient', 'vont', 'votre', 'vous', 'vu', 'y', 'étiez', 'étions', 'été', 'étée', 'étées', 'étés', 'êtes', 'être'
]


def build_output_excel(data):
    if not data:
        print("No differences found between the 2 files.")
        return
        
    df = pd.DataFrame(data, columns=['Notion', 'Occurrences'])
    
    excel_path = '/app/output/excel_results.xlsx'
    # On exporte l'intégralité des notions trouvées dans le fichier Excel
    df.to_excel(excel_path, index=False)
    print(f"Excel file generated: {excel_path}")
    


def build_output_graph(data):
    if not data:
        print("No differences found between the 2 files.")
        return
        
    # Prepares the graph (Top 30)
    df = pd.DataFrame(data[:30], columns=['Notion', 'Occurrences'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Occurrences', y='Notion', data=df, hue='Notion', legend=False, palette='viridis')
    plt.title('Top 30 of notions from file 1 which are missing in file 2')
    plt.tight_layout()
    
    # Saves the image
    output_path = '/app/output/graph_results.png'
    plt.savefig(output_path)
    print(f"Graph generated in {output_path}")


def extract_text(filePath):
    extension = os.path.splitext(filePath)[1].lower()
    
    try:
        if extension == '.txt':
            with open(filePath, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif extension == '.docx':
            doc = Document(filePath)
            # Collects the text of each paragraph and join all the parts of text
            return "\n".join([para.text for para in doc.paragraphs])
        elif extension == '.pdf':
            text = ""
            with fitz.open(filePath) as doc:
                for page in doc:
                    text += page.get_text()
            return text

    except Exception as e:
        print(f"Error reading {filePath}: {e}")
    return ""


def compare_files(filePath1, filePath2):
    print(f"Compare : {os.path.basename(filePath1)} VS {os.path.basename(filePath2)}")

    txt1 = extract_text(filePath1)
    txt2 = extract_text(filePath2)
    
    # Sécurity : if one file is empty
    if not txt1.strip() or not txt2.strip():
        print("One of the files is empty or could not be read.")
        return []
        
    # Scikit-Learn analysis
    vectorizer_params = {
        'stop_words': STOP_WORDS_FR,
        'ngram_range': (1, 2),
        'token_pattern': r"(?u)\b\w{2,}\b", # Mots de 4 lettres minimum
        'max_features': 100,
        'min_df': 1
    }
    vec1 = CountVectorizer(**vectorizer_params)
    try:
        X1 = vec1.fit_transform([txt1])
        words_f1 = vec1.get_feature_names_out()
        counts_f1 = X1.toarray()[0]

        vec2 = CountVectorizer(**vectorizer_params)
        vec2.fit([txt2])
        words_f2 = set(vec2.get_feature_names_out())

        # Filter the notions missing in file 2
        data = [(word, count) for word, count in zip(words_f1, counts_f1) if word not in words_f2]
        data.sort(key=lambda x: x[1], reverse=True)
        
        return data
    except ValueError:
        print("Could not extract enough valid terms for comparison.")
        return []

def start_comparison():
    # Reads the files in input directory
    path_first = '/app/data/first/'
    path_second = '/app/data/second/'
    
    extensions = ("*.txt", "*.docx", "*.pdf")

    files_first = []
    for ext in extensions:
        files_first.extend(glob.glob(os.path.join(path_first, ext)))

    files_second = []
    for ext in extensions:
        files_second.extend(glob.glob(os.path.join(path_second, ext)))
    
    if not files_first or not files_second:
        print(f"Error : a file is required in '{path_first}' and in '{path_second}'")
        return

    file1 = files_first[0]
    file2 = files_second[0]
    
    
    # Launches the comparison
    res = compare_files(file1, file2)
    
    # Builds the excel output file
    build_output_excel(res)
    
    # Builds the output graph
    build_output_graph(res)
    

 
if __name__ == "__main__":
    start_comparison()