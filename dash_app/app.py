
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')

global data
data = pd.read_csv('predicted_outputs.csv', index_col = 0)
data.columns = ['X', 'Y', 'Cluster', 'paper']

#data = pd.DataFrame(np.random.randn(20, 2), columns = ['X', 'Y'])
#print([*['real']*15, *['fake']*5])
#data['paper'] = [*['real']*15, *['fake']*5]


app.layout = html.Div([
				html.Div([
				#Header
				html.H3('Fake paper predictor'), 
				#Input boxes
				html.Label('Input paper title'),
			    dcc.Textarea(id='input_title',
			    placeholder='Enter a paper title',
			    #type='text',
			    value='hello',
			    style={'width': '100%'}
			    ),

			    html.Label('Input paper abstract'),
			    dcc.Textarea(id='input_abstract',
			    placeholder='Enter a the paper abstract',
			    #type='text',
			    value='hello',
			    style={'width': '100%'}
			    ),

			    #Slider to 
			    html.Label('N similar papers'),
			    dcc.Slider(id = 'npaper',
			        min=1,
			        max=10,
			        marks={i: 'Papers {}'.format(i) if i == 1 else str(i) for i in range(1, 11)},
			        value=5),
			    ], style={'columnCount': 1, 'textAlign': 'center','width': '48%'}),
				
				#Submit button
		    html.Div([
    		html.Button('Submit', id='submit-val', n_clicks=0)#,
    		#html.Div(id='container-button-basic',
            # children='Enter a value and press submit')
]),


			    html.Div([
			    dcc.Graph(id='embedding'),
			    #dash_table.DataTable(id = 'table', data = [])

			], style={'columnCount': 1, 'textAlign': 'center', 'width': '48%'}),
	])

@app.callback(
    dash.dependencies.Output('embedding', 'figure'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input_title', 'value'),
    dash.dependencies.State('input_abstract', 'value')
    ]
    )
def update_embedidng(n_clicks, title, abstract):
	if n_clicks == 0:
		traces = []
		for i in data.paper.unique():
			df_by_paper = data[data['paper'] == i]
			traces.append(dict(
	            x=df_by_paper['X'],
	            y=df_by_paper['Y'],
	            #text=df_by_paper['country'],
	            mode='markers',
	            opacity=0.7,
	            marker={
	                'size': 10,
	                'line': {'width': 0.5, 'color': 'white'}
	            },
	            name=i
	        ))
		return {
        	'data': traces,
        	'layout': dict(
            xaxis={'title': 'Component 1'},
            yaxis={'title': 'Component 2'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest',
            transition = {'duration': 500},
        )
    }
	else:
		update_data = transform_input(title, abstract)
		traces = []
		for i in update_data.paper.unique():
			df_by_paper = update_data[update_data['paper'] == i]
			traces.append(dict(
	            x=df_by_paper['X'],
	            y=df_by_paper['Y'],
	            #text=df_by_paper['country'],
	            mode='markers',
	            opacity=0.7,
	            marker={
	                'size': 10,
	                'line': {'width': 0.5, 'color': 'white'}
	            },
	            name=i
	        ))
		return {
        'data': traces,
        'layout': dict(
            xaxis={'title': 'Component 1'},
            yaxis={'title': 'Component 2'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest',
            transition = {'duration': 500},
        )
    }
	   		

def transform_input(title, abstract):
	transformed = np.random.randn(1,2)[0].tolist()
	update_data = pd.concat([data, pd.DataFrame({'X':[transformed[0]],'Y':[transformed[1]],'Cluster':'clust','paper':['Own_paper']})])
	return update_data

#@app.callback(
#    [dash.dependencies.Output("table", "data"), dash.dependencies.Output('table', 'columns')],
#    [dash.dependencies.Input("npaper", "value")]
#)
#def updateTable(value):
#    return data.values[0:10], columns



if __name__ == '__main__':
    app.run_server(debug=True)