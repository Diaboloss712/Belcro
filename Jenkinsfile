pipeline {
    agent any

    environment {
        IMAGE = 'ghcr.io/diaboloss712/belcro:latest'
        CREDENTIALS_ID = 'ghcr-credentials'
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

        stage('Pull Image') {
            steps {
                sh 'docker pull $IMAGE'
            }
        }

        stage('Deploy Container') {
            steps {
                sh '''
                    docker stop belcro || true
                    docker rm belcro || true
                    docker run -d --name belcro -p 8081:80 $IMAGE
                '''
            }
        }
    }
}
