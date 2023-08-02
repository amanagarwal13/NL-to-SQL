import torch
import tensorflow as tf
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import mysql.connector
import spacy
import re
import math
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
    
def cosine_similarity(vector1, vector2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    norm1 = math.sqrt(sum(v1 ** 2 for v1 in vector1))
    norm2 = math.sqrt(sum(v2 ** 2 for v2 in vector2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

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

    keywords = extract_keywords(user_query)
    print(keywords)
    
    similarity_sums = {}
    for table_name in table_names:
        cursor.execute(f"SHOW COLUMNS FROM {table_name};")
        column_names = [column[0] for column in cursor.fetchall()]
        print(column_names)
        
        # Compute cosine similarity between keywords and each column name
        vector1 = [1 if any(keyword.lower() in col_name.lower() for col_name in column_names) else 0 for keyword in keywords]
        vector2 = [1 if any(keyword.lower() in col_name.lower() for col_name in column_names) else 0 for keyword in column_names]
        similarity = cosine_similarity(vector1, vector2)
        print(similarity)
        similarity_sums[table_name]=similarity
    
    max_key = max(similarity_sums, key=lambda k: similarity_sums[k])
    print(similarity_sums)
    print("Identified table name:", max_key)
    return max_key

    cursor.close()
    conn.close()

    max_key = extract_table_name_with_columns(user_query)
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