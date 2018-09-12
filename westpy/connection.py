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
        self.restAPIinit = "http://imedevel.uchicago.edu:8000/getSessionId"
        self.restAPIrun  = "http://imedevel.uchicago.edu:8000/runWestCode"
        self.restAPIstop = "http://imedevel.uchicago.edu:8000/stopSession"
        #
        data = {'emailId': self.emailId ,'sessionTime':str(600)}
        #
        import requests
        import json
        #
        response = None
        try:
            output = requests.post(self.restAPIinit, data=json.dumps(data))
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
    
    def status(self):
        """returns the token to setup the connection.
        
        :Example:
        
        >>> from westpy import *
        >>> connection = Connection("email@domain.com")
        >>> status = connection.status()
        
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
        
        import requests
        import json
        #
        headers = {'Content-Type':'application/json; charset=utf-8','emailId':self.output['emailId'],'token':self.output['token']}
        try:
            response = requests.get(self.restAPIstop, headers=headers, timeout=None)
        except Exception as e:
            print('The Executor is not responding.',e)		 
        return json.loads(response.text) 
    
    def run(self,executable=None,inputFile=None,outputFile=None,downloadUrl=[],number_of_cores=2) :
        """runs the executable using rest api remotely.
        
        :param executable: name of executable
        :type executable: string
        :param inputFile: name of input file
        :type inputFile: string
        :param outputFile: name of output file
        :type outputFile: string
        :param downloadUrl: URLs to be downloaded
        :type downloadUrl: list of string
        :param number_of_cores: number of cores
        :type number_of_cores: int
        
        :Example:
        
        >>> from westpy import *
        >>> connection = Connection("email@domain.com")
        >>> connection.run( "pw", "pw.in", "pw.out", ["http://www.quantum-simulation.org/potentials/sg15_oncv/upf/C_ONCV_PBE-1.0.upf"] , 3 )
        >>> connection.stop()
        
        """
        #
        import json
        #
        output_dict = {}   
        if executable and ("pw" in str(executable).lower() or "wstat" in str(executable).lower() or "wfreq" in str(executable).lower()) :
           # set inputs
           if inputFile is None:
              inputFile = str(executable)+".in"
           if outputFile is None:
              outputFile = str(executable)+".out"		 
           try:
              output = self.__runExecutable(executable,inputFile,downloadUrl,number_of_cores)
              output_json = json.loads(output)
              if "Error" in output_json:
                 print("Execution failed with the following error \n",output_json['Error'])
                 return None			   
              elif "JOB DONE." not in str(output_json['output']).strip():
                 print("MPI execution failed with the following error:  \n"+str(output))
                 return None
              output_data = str(output_json['output']).strip()
              # jasonify output
              if "pw" in executable:
                  output_dict = json.loads(output_json['output_dict'])
              else:
                  output_dict = output_json['output_dict']
              # write the output file
              with open(outputFile, "w") as file :
                 file.write(str(output_data))
           except Exception as e:
              print("Session Expired! Invalid Request sent, Please recreate session and recheck your input. \n"+ e)		 
              return None          
        else:
           raise ValueError("Invalid Executable name") 
        #
        print("Generated ",outputFile)
        return output_dict
      
    def __runExecutable(self,executable,input_file,download_urls,number_of_cores) :
        """Runs remotely the executable using a REST api.
        """
        #
        import requests
        import json
        # get connection output
        session_values = self.status()
        # suck in the input file
        try:
           file_content = ""	     
           with open(input_file,'r') as f :
              for line in f :
                 file_content = file_content + line + "\\n"
        except FileNotFoundError:
           error = "Could not find "+ input_file + ". \n Generate input file "+ input_file +" and try again."	  
           print(error)
           return None
        body = {'urls':download_urls,'file':file_content,'cmd_timeout':'600','script_type':str(executable),'no_of_cores':str(number_of_cores)}  
        jsondata = json.dumps(body)
        jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
        headers = {'Content-Type':'application/json; charset=utf-8','emailId':session_values['emailId'],'token':session_values['token']}
        response = ""
        try:
           response = requests.post(self.restAPIrun, data = jsondataasbytes, headers=headers, timeout=None)
           return response.text
        except Exception as e:
           print('The Executor is not responding.',e)	
           return None
        return None
