# Load Balancer - 5xx Spike

## Symptoms
- Sudden increase in 502/503/504
- Backend health checks failing
- Latency increase at LB

## Initial Checks
1. Check backend target health
2. Check recent deploys on backend services
3. Verify LB config changes and health check path

## Remediation
- Drain bad targets
- Roll back recent config changes
- Scale backend service if saturated

## Escalation
- Page platform/LB team if unresolved after 15 minutes

