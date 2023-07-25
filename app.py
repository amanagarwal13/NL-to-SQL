import torch
import tensorflow as tf
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import mysql.connector
import spacy
import re
from flask import Flask,render_template,url_for,request

output_dir = './fine-tuned-model'

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(output_dir)
device = torch.device('cuda' if tf.test.is_gpu_available() else 'cpu')
model=model.to(device)

def remove_tags(text):
    text = text.replace("<pad>", "").replace("</s>", "")
    return text.strip()

def extract_keywords(user_query):

    nlp = spacy.load("en_core_web_sm")

    doc = nlp(user_query)

    keywords = [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN", "ADJ"}]

    return keywords



def nlp(input_question):
    
        input_encoded = tokenizer.encode_plus(
        input_question,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
        ).to(device)

        generated = model.generate(
        input_ids=input_encoded.input_ids,
        attention_mask=input_encoded.attention_mask,
        max_length=64,
        num_beams=4,
        early_stopping=True
        )
        
        generated_query = tokenizer.decode(generated.squeeze())
        generated_query = remove_tags(generated_query)
        return generated_query
    
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def extract_table_name_with_columns(user_query):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Aman@7221",
        database="world"
    )
    cursor = conn.cursor()

    cursor.execute("SHOW TABLES;")
    table_names = [table[0] for table in cursor.fetchall()]

    table_columns_mapping = {}

    keywords = extract_keywords(user_query)
    #print(keywords)

    similarity_sums = {}
    for table_name in table_names:
        cursor.execute(f"SHOW COLUMNS FROM {table_name};")
        column_names = [column[0] for column in cursor.fetchall()]
        #print(column_names)

        # Compute Jaccard similarity between keywords and each column name
        similarities = [jaccard_similarity(set(keywords), set(column_name.lower().split())) for column_name in column_names]
        #print(similarities)
        sum_similarity=sum(similarities)
        similarity_sums[table_name]=sum_similarity

        # Find the column name with the highest similarity score
        max_similarity_index = similarities.index(max(similarities))
        max_similarity_score = similarities[max_similarity_index]

        threshold = 0.1
        if max_similarity_score >= threshold:
            table_columns_mapping[table_name] = column_names[max_similarity_index]

    #print(table_columns_mapping)
    #print(similarity_sums)

    max_key = max(similarity_sums, key=lambda k: similarity_sums[k])

    print("Identified table name:", max_key)
    print("Associated column name:", table_columns_mapping[max_key])
    return(max_key)


def display_table(user_query):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Aman@7221",
        database="world"
    )
    cursor = conn.cursor()
    cursor.execute(user_query)
    results = cursor.fetchall()
    conn.commit()
    conn.close()
    return results

def change_next_word(text, particular_word, new_word):
    words = text.split()
    for i, word in enumerate(words[:-1]):
        if word == particular_word:
            words[i + 1] = new_word
    return ' '.join(words)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the user's input from the form
        user_input = request.form['user_input']
        data=str(user_input)
        sql_query=nlp(data)
        words = re.findall(r'=\s*(\w+)', sql_query)
        for word in words:
            sql_query = re.sub(rf'= {word}', f'= \'{word}\'', sql_query)   

        max_key=extract_table_name_with_columns(sql_query)
        
        max_key=str(max_key)
        #print(max_key)
        modified_text = change_next_word(sql_query, "FROM", max_key)
        
        results=display_table(modified_text)
        
        #max_records = 15
        #limited_results = results[:max_records]
        # Process the input (you can add your own processing logic here if needed)

        # Pass the processed input to the template to display below the form
        return render_template('home.html', output=modified_text, results=results )
    #results=results 
    # If the request method is GET, display an empty form
    return render_template('home.html', user_input='')

if __name__ == '__main__':
    app.run(debug=True)