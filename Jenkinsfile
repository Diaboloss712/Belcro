pipeline {
    agent any

    environment {
        IMAGE_NAME = 'ghcr.io/diaboloss712/belcro'
        TAG = "latest" // or use "${env.BUILD_NUMBER}" for unique tagging
        FULL_IMAGE = "${IMAGE_NAME}:${TAG}"
        CREDENTIALS_ID = 'ghcr-credentials'
        UPSTAGE_API_KEY = credentials('UPSTAGE_API_KEY')
        PINECONE_API_KEY = credentials('PINECONE_API_KEY')
    }

    triggers {
        githubPush()
    }

    stages {
        stage('Checkout') {
            steps {
                git credentialsId: 'ghcr-credentials', url: 'https://github.com/Diaboloss712/Belcro'
            }
        }

        stage('Login to GHCR') {
            steps {
                withCredentials([usernamePassword(credentialsId: env.CREDENTIALS_ID, usernameVariable: 'USER', passwordVariable: 'TOKEN')]) {
                    sh '''
                        echo "$TOKEN" | docker login ghcr.io -u "$USER" --password-stdin
                    '''
                }
            }
        }

        stage('Build and Push Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${IMAGE_NAME}", ".")
                    dockerImage.push("${TAG}")
                }
            }
        }

        stage('Deploy Container') {
            steps {
                sh '''
                    docker stop belcro || true
                    docker rm belcro || true
                    docker run -d --name belcro -p 8081:80 \
                        -e UPSTAGE_API_KEY=$UPSTAGE_API_KEY \
                        -e PINECONE_API_KEY=$PINECONE_API_KEY \
                        $FULL_IMAGE
                '''
            }
        }
    }
}
