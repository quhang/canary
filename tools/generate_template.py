#!/usr/bin/env python
import string
import sys
import os.path
kProjectName = 'canary'
kCopyrightTemplate = '''\
/*
 * Copyright 2015 Stanford University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither the name of the copyright holders nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
'''
kHeaderTemplate = '''\
/**
 * @file {file_name}
 * @author {author_name}
 * @brief Class {class_name}.
 */
'''

def GenerateTemplate(file_name):
    if os.path.lexists(file_name):
        print "File already exists."
        return
    if not os.path.isdir(os.path.dirname(file_name)):
        print "Directory does not exist."
        return
    normal_name = os.path.normpath(file_name)
    dir_name = os.path.dirname(normal_name)
    base_name = os.path.basename(normal_name)
    name, extension = os.path.splitext(base_name)
    assert extension in {'.h', '.cc'},\
            'Do not support extension: {}'.format(extension)
    macro_name = '{}_{}_H_'.format(
            kProjectName.upper(),
            (dir_name+'/'+name).replace('/','_').replace('-','_').upper())
    class_name = name.title().translate(None, '_-')
    code = open(file_name, 'w')
    code.write(kCopyrightTemplate)
    code.write(kHeaderTemplate.format(
        file_name=normal_name,
        author_name = 'Hang Qu (quhang@cs.stanford.edu)',
        class_name = class_name))
    code.write('\n')
    if extension == '.h':
        code.write('#ifndef {}\n#define {}\n'.format(macro_name, macro_name))
        code.write('namespace {} {{\n\n'.format(kProjectName))
        code.write('class {} {{\n'.format(class_name))
        code.write('}};\n\n'.format(class_name))
        code.write('}}  // namespace {}\n'.format(kProjectName))
        code.write('#endif  // {}\n'.format(macro_name))
    else:
        code.write('namespace {} {{\n\n\n'.format(kProjectName))
        code.write('}}  // namespace {}\n'.format(kProjectName))
    code.close()

if __name__ == '__main__':
    GenerateTemplate(sys.argv[1])
