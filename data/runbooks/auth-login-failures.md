# Authentication - Login Failures

## Symptoms
- Users can’t log in
- Increased 401/403
- Spikes in auth-service 5xx

## Initial Checks
1. Check auth-service health endpoint
2. Verify database connectivity
3. Check recent deployments

## Remediation
- Restart auth-service
- Roll back last deployment if needed

## Escalation
- Page identity team if unresolved after 15 minutes

