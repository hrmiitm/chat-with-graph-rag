# Custom PostgreSQL image with pgvector + Apache AGE extensions
# Provides both vector similarity search and graph database capabilities

FROM postgres:16

# Build dependencies for compiling extensions from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    postgresql-server-dev-16 \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector — vector similarity search extension
RUN git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Install Apache AGE — graph database extension
RUN git clone --branch PG16/v1.5.0-rc0 https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make \
    && make install \
    && rm -rf /tmp/age

# Preload AGE shared library (required for Cypher query support)
RUN echo "shared_preload_libraries = 'age'" >> /usr/share/postgresql/postgresql.conf.sample

# Copy initialization script — runs on first container start
COPY init.sql /docker-entrypoint-initdb.d/01-init.sql
