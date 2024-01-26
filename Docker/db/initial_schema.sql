
-- Create the task table with a manually managed sequence
CREATE TABLE public.task (
    task_template TEXT NOT NULL
);

-- Create the resource table with a manually managed sequence
CREATE TABLE public.resource (
    task_template TEXT NOT NULL
);
