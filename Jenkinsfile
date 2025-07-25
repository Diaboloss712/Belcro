pipeline {
    agent any

    environment {
        IMAGE_NAME = 'ghcr.io/diaboloss712/belcro'
        TAG = 'latest'
        FULL_IMAGE = "${IMAGE_NAME}:${TAG}"
        CREDENTIALS_ID = 'ghcr-credentials'
    }

    triggers {
        githubPush()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Docker Build & Push') {
            environment {
                REGISTRY = 'https://ghcr.io'
            }
            steps {
                script {
                    docker.withRegistry(REGISTRY, CREDENTIALS_ID) {
                        def image = docker.build(IMAGE_NAME)
                        image.push(TAG)
                    }
                }
            }
        }

        stage('Deploy') {
            environment {
                UPSTAGE_API_KEY = credentials('UPSTAGE_API_KEY')
                PINECONE_API_KEY = credentials('PINECONE_API_KEY')
            }
            steps {
                script {
                    sh """
                        docker stop belcro || true
                        docker rm belcro || true

                        docker run -d --name belcro -p 8081:80 \\
                            -e UPSTAGE_API_KEY=${UPSTAGE_API_KEY} \\
                            -e PINECONE_API_KEY=${PINECONE_API_KEY} \\
                            ${FULL_IMAGE}
                    """
                }
            }
        }
    }
}
