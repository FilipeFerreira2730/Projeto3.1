CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS POSTGIS;
CREATE EXTENSION IF NOT EXISTS POSTGIS_TOPOLOGY;


-- Create a sequence for idTask
CREATE SEQUENCE task_id_seq;

-- Create the task table with a manually managed sequence
CREATE TABLE public.task (
    idTask INTEGER PRIMARY KEY DEFAULT nextval('task_id_seq'),
    task_template TEXT NOT NULL
);

-- Create a sequence for idResource
CREATE SEQUENCE resource_id_seq;

-- Create the resource table with a manually managed sequence
CREATE TABLE public.resource (
    idResource INTEGER PRIMARY KEY DEFAULT nextval('resource_id_seq'),
    task_template TEXT NOT NULL
);
