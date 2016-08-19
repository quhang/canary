import socket
import subprocess
import sys

def is_valid_hostname(hostname):
    try:
        socket.getaddrinfo(hostname, None)
    except socket.gaierror as err:
        print 'Cannot connect to {}: {}'.format(hostname, err)
        return False
    else:
        return True

def run_on_all_workers(filename, command_template):
    status_dict = dict()
    with open(filename, 'r') as handle:
        for line in handle:
            hostname = line.split()[0]
            assert is_valid_hostname(hostname)
            status = subprocess.Popen(
                    command_template.format(hostname),
                    shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
            assert hostname not in status_dict, 'Duplicate host names!'
            status_dict[hostname] = status
    while status_dict:
        for hostname in status_dict.keys():
            if status_dict[hostname].poll() != None:
                print '{} returns {}'.format(hostname,
                                             status_dict[hostname].returncode)
                del status_dict[hostname]

def start_workers(filename, controller_ip):
    command_template = '''ssh \
            -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
            ubuntu@{} \
            /home/ubuntu/build/canary_worker \
            --worker_threads=8 --controller_host={}'''.format('{}',
                                                              controller_ip)
    run_on_all_workers(filename, command_template)

def stop_workers(filename):
    command_template = '''ssh \
            -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
            ubuntu@{} killall canary_worker'''
    run_on_all_workers(filename, command_template)

def print_usage_message(program_name):
    print "usage: {} start/stop filename controller_ip".format(program_name)

if __name__ == '__main__':
    if len(sys.argv) not in {3, 4}:
        print_usage_message(sys.argv[0])
        exit()
    if sys.argv[1] == 'start':
        if len(sys.argv) != 4:
            print_usage_message(sys.argv[0])
            exit()
        start_workers(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'stop':
        if len(sys.argv) != 3:
            print_usage_message(sys.argv[0])
            exit()
        stop_workers(sys.argv[2])
    else:
        print_usage_message(sys.argv[0])
        exit()
