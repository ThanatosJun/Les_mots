import streamlit as st
import ollama

def main():
    st.title("第一階段-建立畫面與測試原模型")
    user_input = st.text_area("Input your question:")
    if st.button("Submit"):
        if user_input:
            response = ollama.chat(
                model = "llama3.2",
                messages = [{"role": "user", "content": user_input}],
                )
            st.text("Response:")
            st.write(response["message"]["content"])
        else:
            st.warning("Please Enter Your Questions")

if __name__ == "__main__":
    main()