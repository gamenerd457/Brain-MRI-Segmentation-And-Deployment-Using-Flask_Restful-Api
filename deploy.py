from flask import Flask
from flask_restful import Resource,Api,reqparse
from infer import infer
from werkzeug.datastructures import FileStorage
import tempfile

app=Flask(__name__)
app.logger.setLevel('INFO')
api=Api(app)

parser=reqparse.RequestParser()
parser.add_argument('file',required=True,
	location='files',
	type=FileStorage,
	help="provide an image file")

class Image(Resource):
	def post(self): #creating a post method 
		args = parser.parse_args()
		the_file = args['file']
		#saving a temp copy of the file
		ofile, ofname = tempfile.mkstemp()
		the_file.save(ofname)
		infer(ofname) #running the inference

#creating the /image  endpoint
api.add_resource(Image,'/image') 
if __name__ == '__main__':
    app.run(debug=True)		

