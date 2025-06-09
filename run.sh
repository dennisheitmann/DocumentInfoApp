#!/bin/sh
streamlit run app.py --server.port 9999 --server.baseUrlPath=/pdfinfo/ --browser.gatherUsageStats=false --server.maxUploadSize=10
