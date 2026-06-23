make run:
	uv run python assistant.py 

chat: 
	uv run streamlit run app.py

network: 
	./create_docker_network.sh

# postgres: network
postgres: 
	./run_container.sh

postgres-rerun: 
	docker start course-assistant-pg 

postgres-rerun-it: 
	docker exec -it course-assistant-pg bash 
# docker exec -it course-assistant-pg sh  

stop-postgres: 
	docker stop course-assistant-pg 

remove-postgres: 
	docker rmi postgres:17

remove-postgres-volume: 
	docker volume rm pgdata 

check-db: 
	docker exec -it course-assistant-pg psql -U user -d course_assistant \
    -c "SELECT id, question, response_time, cost FROM conversations;"

query: 
	uv run python db_query.py 

dashboard: 
	uv run streamlit run dashboard.py --server.port 8502

generate-data:
	uv run python generate_data.py

grafana: 
	./run_grafana.sh 

stop-grafana:
	docker stop grafana 

remove-grafana: 
	docker rmi grafana/grafana:latest 

remove-grafana-volume: 
	docker volume rm grafana_data 

remove-all-volumes: 
	docker rm -f $(docker ps -aq) && docker volume rm $(docker volume ls -q)


