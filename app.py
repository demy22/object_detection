import streamlit as st
import openai
import keras_ocr
import cv2


openai.api_key = st.secrets["api_key"]
data_dir = ''
image_url = ''
st.title("GPT + DALL-E & Object Detection")


detector = keras_ocr.detection.Detector()
recognizer = keras_ocr.recognition.Recognizer()


        # restore model weights
detector_model_path = os.path.join(data_dir, 'detector_carplate.h5')
recognizer_model_path = os.path.join(data_dir, 'recognizer_carplate.h5')
if os.path.isfile(detector_model_path) == True:
    detector.model.load_weights(detector_model_path)
    recognizer.model.load_weights(recognizer_model_path)
    #recognizer.compile()
    print('pre-trained model loaded!')
recognizer.compile()



with st.form("form"):
    user_input = st.text_input("Prompt")
    size = st.selectbox("Size", ["1024x1024", "512x512", "256x256"])
    submit = st.form_submit_button("이미지 생성 & 검출")

if submit and user_input:
    gpt_prompt = [{
        "role": "system",
        # "content" : "You are a helpful assistant"
        "content": "Imagine the detail\
        appeareance of the input. Response it within 100 words"
    }]

    gpt_prompt.append({
        "role": "user",
        "content": user_input
    })

    msg = ''


    with st.spinner("Waiting for ChatGPT..."):
        gpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                    messages=gpt_prompt)
    
        msg = gpt_response["choices"][0]["message"]["content"]
        st.write(msg)
        
        
        gpt_prompt.append({
            "role": "assistant",
            "content": msg
        })

        gpt_prompt.append({
            "role": "user",
            "content": 'Translate to english'
        })
        
        gpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                    messages=gpt_prompt)
    

        msg = gpt_response["choices"][0]["message"]["content"]
        st.write(msg)
        
        # gpt_prompt.append({
        #     "role": "assistant",
        #     "content": msg
        # })

        # gpt_prompt.append({
        #     "role": "user",
        #     "content": 'Condense the description to focus on nouns and adjectives, and separated by ,'
        # })

        # gpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        #                             messages=gpt_prompt)
        
        # msg = gpt_response["choices"][0]["message"]["content"]
        # st.write(msg)

    with st.spinner("Waiting for DALL-E..."):
        dalle_response = openai.Image.create(
            prompt=msg,
            size=size
        )

    image_url = dalle_response["data"][0]["url"]
    st.image(dalle_response["data"][0]["url"])

    
    image = keras_ocr.tools.read(image_url)


    pred_boxes = detector.detect(np.expand_dims(image, axis=0))
    for each_pred in pred_boxes[0]:
        left, top = each_pred[0]
        right, bottom = each_pred[2]
        image = cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0,255,0), 3)

    #pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)
    #pipeline = keras_ocr.pipeline.Pipeline()

    prediction = recognizer.recognize(image)
    print(prediction)
    #keras_ocr.tools.drawAnnotations(image=image, predictions=prediction)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'predict : {prediction}', (10,20), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    st.image(image)
    
    st.write("============Detection Test============")
    
    image = keras_ocr.tools.read('car1.jpg')
    st.image(image)
    pred_boxes = detector.detect(np.expand_dims(image, axis=0))
    for each_pred in pred_boxes[0]:
        left, top = each_pred[0]
        right, bottom = each_pred[2]
        image = cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0,255,0), 3)

    st.image(image)
    st.write("============OCR Test============")

    image = keras_ocr.tools.read('al47.png')
    st.image(image)
    prediction = recognizer.recognize(image)
    print(prediction)
    #keras_ocr.tools.drawAnnotations(image=image, predictions=prediction)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'predict : {prediction}', (10,20), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    st.image(image)



