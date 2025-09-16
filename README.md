# Instructions to install

Install packages from requirements.txt:
pip install -r requirements.txt

Download the SSL certificate from TiDB:
Connect > Connect with SQLAlchemy > Download CA Cert
Rename it to cert.pem
Paste it in the same folder as the program

Create a .env file like so:
TIDB_USER=<your_username>
TIDB_PASSWORD=<your_password>
TIDB_HOST=<your_host>
TIDB_PORT=4000
TIDB_DATABASE=test
OPENAI_API_KEY=<your_openai_api_key>

Run the program:
streamlit run main.py