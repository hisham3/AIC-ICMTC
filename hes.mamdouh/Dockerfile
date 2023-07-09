FROM aic1mtc/aic-1-mtc-competition:0.1
WORKDIR /submission
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN mkdir /submission/src
COPY src ./src
COPY infere.py ./
COPY infere.sh ./
CMD [ "/bin/bash","infere.sh" ]
