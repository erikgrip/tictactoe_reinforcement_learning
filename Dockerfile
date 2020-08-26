FROM python:3.8-slim

WORKDIR /tic_tac_toe

VOLUME /tic_tac_toe

CMD ["python", "../tic_tac_toe"]

