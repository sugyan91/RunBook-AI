# Network - DNS Resolution Issues

## Symptoms
- Intermittent timeouts
- Services can’t resolve hostnames
- Errors like "NXDOMAIN" or "Temporary failure in name resolution"

## Initial Checks
1. Check DNS resolver health (internal/external)
2. Validate `/etc/resolv.conf` or node DNS config
3. Test resolution from affected hosts (dig/nslookup)

## Remediation
- Restart DNS caching service if used
- Fail over to secondary resolver
- Roll back network policy changes if applicable

## Escalation
- Page network/on-call if unresolved after 15 minutes

