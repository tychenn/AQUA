from google import genai
client = genai.Client(api_key="YOUR-API-KEY")
def calculate_simscore(txt1:str,txt2:str)->str:
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-05-06", contents=f"Determine the semantic similarity between the following two strings and \
            give your score on a scale of 0-100: \
            String 1: {txt1}\n\
            String 2: {txt2}\n\
            Just answer with numbers."
    )   
    return response