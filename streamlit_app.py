# import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# import numpy as np
import streamlit as st


st.set_page_config(layout="wide")
fs = 18

xvar = 'durata'
# yvar = 'interessi'
yvar = 'cedola'
# cvar = 'cedola'
cvar = 'prezzo'

import pandas as pd

# URL of the webpage containing the table
valuta='EUR'
monitor='europa'
monitor='italia'
monitor = 'sovranazionali'
monitor='banche'
# monitor='bancheitalia'
# monitor='corporate'
monitor='corporateitalia'
interessi='G' # lordo
# interessi='N' # netto
volume=0

monitor_list = ('europa', 'italia', 'sovranazionali', 'banche', 'bancheitalia', 'corporate', 'corporateitalia')


c1, c2, c3  = st.columns(3)

with c1:
    monitor = st.selectbox(
    "Scegli tipo di Bond",
    monitor_list,
)

with c2:
    intr = st.selectbox(
        "Scegli tipo di interessi",
        ('Lordo', 'Netto')
    )

    interessi = 'G' if intr == 'Lordo' else 'N'

with c3:
    yvar = st.selectbox(
        "Scegli asse Y",
        ('Interessi', 'Cedola')
    )

    yvar = yvar.lower()



yaxis= f'Interessi {"netti" if interessi == "N" else "lordi"} [%]'


url = f'https://www.simpletoolsforinvestors.eu/monitor_info.php?monitor={monitor}&timescale=DUR&yieldtype={interessi}&currency={valuta}&volumerating={volume}'


get_cedola = lambda x: x.split('%')[0].split(' ')[-1]

def parse_cedola(x):
    # print(x)
    try:
        return float(x.replace(',', '.'))
    except:
        return 0


if 'data' not in st.session_state: 
    st.session_state.data = {'N':{mm: [] for mm in monitor_list}, 'G':{mm: [] for mm in monitor_list}}

# Check if any tables were found
if len(st.session_state.data[interessi][monitor]) == 0:
    # Read the table from the HTML page by specifying the table's id
    tables = pd.read_html(url, attrs={'id': 'YieldTable'})
    print('Downloaded',len(tables),'tables')
    if tables:
        # The result is a list of DataFrames; extract the first one
        monitorDF = tables[0]
        print('Parsed',len(monitorDF),'rows')
        # print(monitorDF.columns)
        monitorDF.loc[:, 'prezzo'] = monitorDF['Prezzo diriferimento'] / 1e2
        monitorDF.loc[:, 'interessi'] = monitorDF['Yield'] / 1e2
        monitorDF.loc[:, 'durata'] = monitorDF['Duration'] / 1e2
        monitorDF.loc[:, 'cedola'] = monitorDF.Descrizione.apply(get_cedola)
        monitorDF.loc[:, 'cedola'] = monitorDF.cedola.apply(parse_cedola)
        monitorDF.loc[:, 'volume'] = monitorDF['Volume(Milioni)']/1e3
        st.session_state.data[interessi][monitor] = monitorDF.copy()
        st.success(f'Dati scaricati e per monitor = {monitor.capitalize()} e interessi {interessi}', icon="ℹ️")
    else:
        print("No table with the specified id found.")
else:
    print(f'Data already downlaoded for monitor = {monitor.capitalize()} e interessi {interessi}')
    st.info(f'Dati gia scaricati per monitor = {monitor.capitalize()} e interessi {interessi}', icon="ℹ️")
    monitorDF = st.session_state.data[interessi][monitor].copy()

# mask = (monitorDF['volume'] >= 0 ) & (monitorDF['prezzo'] < 99 ) & (monitorDF['durata'] < 11) & (monitorDF['interessi'] >= 2) # & (monitorDF['interessi'] <= 5)
# mask = (monitorDF['volume'] >= 0 ) & (monitorDF['prezzo'] < 99 ) & (monitorDF['durata'] < 11) # & ((monitorDF['interessi'] >= 2) | (monitorDF['cedola'] > 0.5)) # & (monitorDF['interessi'] <= 5)
pdf = monitorDF.copy()#[mask].copy()
x = pdf[xvar] # np.random.rand(100)
y = pdf[yvar] #np.random.rand(100)
clr = pdf[cvar]

# Streamlit app
# st.title('Interactive Scatter Plot with Thresholds')

# # Slider for X-axis threshold
# x_threshold = st.slider('Select X-axis threshold', min_value=float(0), max_value=float(monitorDF[xvar].max()), value=float(monitorDF[xvar].max()))

# # Slider for Y-axis threshold
# y_threshold = st.slider('Select Y-axis threshold', min_value=float(0), max_value=float(monitorDF[yvar].max()), value=float(monitorDF[yvar].max()))




intm, cedm, volm, przM = 0, 0, 0, 0


durvalues = st.slider(
    'Scegli range di valori per durata [anni]',
    0.00, float(monitorDF[xvar].max()), (0.00, float(monitorDF[xvar].max())))
durm, durM = durvalues

c1, c2 = st.columns(2)
with c1:
    intm = st.slider(
        f'Scegli minimo per '+yaxis.lower(),
        0.00, float(monitorDF[yvar].max()), 0.00)
    # intm, intM = intvalues

    cedm = st.slider(
        'Scegli minimo per cedola [%]',
        0.00, float(monitorDF['cedola'].max()), 0.00)
    # cedm, cedM = cedvalues

with c2:
    volm = st.slider(
        'Scegli minimo per volume [milioni]',
        0.00, float(monitorDF['volume'].max()), 0.00)
    

    przM = st.slider(
        'Scegli massimo per prezzo [euro]',
        0.00, float(monitorDF['prezzo'].max()), 99.00)
    # przm, przM = przvalues






# vol_threshold = st.slider('Select Volume threshold', min_value=(float(0), float(monitorDF['volume'].max())), max_value=(float(0), float(monitorDF['volume'].max())), step=(0.01, 0.01))

# Text box for custom threshold
# custom_threshold = st.text_input('Enter custom threshold', value=str(float(x.mean())))

# Filter data based on thresholds
# filtered_data = customdata[(x >= x_threshold) & (y >= y_threshold)]

# Update scatter plot with filtered data
# fig.data[0].x = filtered_data[:, 0]
# fig.data[0].y = filtered_data[:, 1]
# fig.data[0].marker.size = filtered_data[:, 3]
# fig.data[0].marker.color = filtered_data[:, 2]
# fig.data[0].customdata = filtered_data

# mask = (monitorDF['volume'] >= 0 ) & (monitorDF['prezzo'] < 99 ) & (monitorDF['durata'] < 11) & (monitorDF['interessi'] >= 2) # & (monitorDF['interessi'] <= 5)
mask = (monitorDF['volume'] >= volm ) & (monitorDF['prezzo'] <= przM ) & (monitorDF['durata'] >= durm) & (monitorDF['durata'] <= durM) & (monitorDF['interessi'] >= intm) & (monitorDF['cedola'] >= cedm) # & (monitorDF['interessi'] <= 5)
pdf = monitorDF[mask].copy()

# print(len(pdf))

# Generate sample data
# np.random.seed(42)
x = pdf[xvar] # np.random.rand(100)
y = pdf[yvar] #np.random.rand(100)
clr = pdf[cvar]
# clr = pdf.prezzo

# marker_sizes = (pdf.cedola+2) * 4
# marker_sizes = (pdf.interessi+1) * 4
marker_sizes = np.log((pdf.volume+1) * 1e4)


customdata = [[dataScad, cedola, ISINcode, descr, prezzo, volume, interessi] for dataScad, cedola, ISINcode, descr, prezzo, volume, interessi in zip(pdf.Datascadenza, pdf.cedola, pdf['Codice ISIN'], pdf.Descrizione, pdf.prezzo, pdf.volume, pdf.interessi)]
# sum_xy = x + y



st.markdown(f'## Monitor {monitor.capitalize()}')
# Create the scatter plot
fig = go.Figure(data=go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        size=marker_sizes,
        color=clr,
        # colorscale='viridis',
        # colorscale='plasma_r',
        colorscale='Reds',
        showscale=True,
        colorbar=dict(
            title=cvar.capitalize(),
            titlefont=dict(size=fs + 4),
            tickfont=dict(size=fs)
        )
    ),
    hovertemplate=(
        'Durata: %{x:.2f} (%{customdata[0]}) <br>' +
        'Cedola:%{customdata[1]}<br>' +
        yaxis.replace(' [%]', '') + ': %{customdata[6]:.2f}<br>' +
        'ISIN: %{customdata[2]}<br>' +
        '%{customdata[3]}<br>-----<br>' +
        'Prezzo:%{customdata[4]}<br>' +
        'Volume: %{customdata[5]}' + '<extra></extra>'
    ),
    customdata=customdata,
))

fig.update_layout(
    # title=monitor.capitalize(),
    xaxis_title='Durata [anni]',
    yaxis_title=yaxis,
    # yaxis_title=yvar.capitalize() + ' [%]',
    xaxis=dict(title_font=dict(size=fs + 4), tickfont=dict(size=fs)),
    yaxis=dict(title_font=dict(size=fs + 4), tickfont=dict(size=fs)),
    width=13600,
    height=600
)


# Display the updated plot
st.plotly_chart(fig)


st.text(f'Numero titoli = {len(pdf)}')
pdf.loc[:, f'interessi {interessi}'] = pdf.interessi
# st.table(pdf[['Codice ISIN', 'Descrizione', 'Divisa', 'Datascadenza', 'Lottominimo', 'Status', 'Mercato', 'TipoCalcolo', 'prezzo',f'interessi {interessi}','durata','cedola', 'volume']].sort_values(by='durata', ascending=True))
st.dataframe(pdf[['Codice ISIN', 'Descrizione', 'Divisa', 'Datascadenza', 'Lottominimo', 'Status', 'Mercato', 'TipoCalcolo', 'prezzo',f'interessi {interessi}','durata','cedola', 'volume']].sort_values(by='durata', ascending=True),
hide_index=True,
use_container_width=True
)
# st.table(pdf)
