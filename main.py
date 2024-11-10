import base64
import streamlit as st

with open('assets/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.logo("assets/logo_sbsgm.png")
st.html("""
  <style>
    [alt=Logo] {
      height: 12rem;
      text-align:center;
    }
  </style>
        """)
st.sidebar.title("Sağlık Bakanlığı AutoML")
def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    return html

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.image("assets/logo_sb.png", caption="AutoML Sistemi")
    username = st.text_input("Kullanıcı Adı", key="username")
    password = st.text_input("Parola", key="password", type="password")
    if st.button("Giriş") and username == "test" and password == "test":
        st.session_state.logged_in = True
        st.rerun()
    st.divider()
    cols = st.columns(2) 
    cols[0].write(render_svg(open("assets/OrtakGirisLogo.svg").read()), unsafe_allow_html=True)
    cols[1].image('assets/e-devlet-logo.png', use_container_width=True) 
   
    
    

def logout():
    if st.button("Çıkış"):
        st.session_state.logged_in = False
        st.rerun()

login_page = st.Page(login, title="Giriş", icon=":material/login:")
logout_page = st.Page(logout, title="Çıkış", icon=":material/logout:")


regression = st.Page("pages/regression.py", title="Regresyon Analizi", icon=":material/bug_report:")
classification = st.Page("pages/classification.py", title="Sınıflandırma", icon=":material/bug_report:")
clustering = st.Page("pages/clustering.py", title="Kümeleme Analizi", icon=":material/bug_report:")
anomaly = st.Page("pages/anomaly.py", title="Anomali Analizi", icon=":material/bug_report:")
dimension = st.Page("pages/dimension.py", title="Boyut Analizi", icon=":material/bug_report:")
image_cls = st.Page("pages/ataberk.py", title="Ataberk Analizi", icon=":material/bug_report:")


if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Hesap": [logout_page],
            "Modeller": [regression, classification, clustering,anomaly,dimension, image_cls],
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()