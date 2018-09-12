from __future__ import print_function

class Connection(object):
	"""Class for setting up the Connection to the Rest API Server.

	:Example:

	>>> from westpy import *
	>>> connection = Connection("email@domain.com")

	"""
	def __init__(self,emailId) :
		self.output = {}
		self.emailId = str(emailId)
                self.restAPIgetSessionId = "http://imedevel.uchicago.edu:8000/getSessionId"
                self.restAPIstopSession = "http://imedevel.uchicago.edu:8000/stopSession"
                #
		data = {'emailId': self.emailId ,'sessionTime':str(600)}
                #
                import requests
                import json
                #
		response = None
		try:
			output = requests.post(self.restAPIgetSessionId, data=json.dumps(data))
			response = json.loads(output.text)
		except Exception as e:
			print('The Executor is not responding.',e)
		if response:
			if "Error" in response:
				print("Execution failed with the following error \n",response['Error'])
				return None
			else:
				print("Check the inbox/spam folder of your email and click on the link to activate the session")
				self.output = response
		else:
			print('The Executor is not responding.')
	
	
	def getOutput(self):
		"""returns the token to setup the connection.

		:Example:

		>>> from westpy import *
		>>> connection = Connection("email@domain.com")
                >>> connection.getOutput()

		"""
		if self.output:
			return self.output
		else:
			raise ValueError("Cannot find output.")

	def stop(self):
		"""stops the executable using the token

		:Example:

		>>> from westpy import *
		>>> connection = Connection("email@domain.com")
		>>> connection.stop()

		"""

                #
                import requests
                import json
                #
		headers = {'Content-Type':'application/json; charset=utf-8','emailId':self.output['emailId'],'token':self.output['token']}
		try:
			response = requests.get(self.restAPIstopSession, headers=headers, timeout=None)
		except Exception as e:
			print('The Executor is not responding.',e)		 
		return json.loads(response.text)      
