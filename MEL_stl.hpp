/*
The MIT License(MIT)

Copyright(c) 2016 Joss Whittle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

#include "MEL.hpp"

#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include <iterator>

namespace MEL {
	
	template<typename It>
	inline void Send(It first, It last, const Datatype &datatype, const int dst, const int tag, const Comm &comm, std::random_access_iterator_tag) {
		MEL::Send(&(*first), std::distance(first, last), datatype, dst, tag, comm);
	};

	template<typename It>
	inline void Send(It first, It last, const Datatype &datatype, const int dst, const int tag, const Comm &comm) {
		typedef typename std::iterator_traits<It>::iterator_category category;
		MEL::Send(first, last, datatype, dst, tag, comm, category());
	};

	template<typename It>
	inline void Recv(It first, It last, const Datatype &datatype, const int src, const int tag, const Comm &comm, std::random_access_iterator_tag) {
		MEL::Recv(&(*first), std::distance(first, last), datatype, src, tag, comm);
	};

	template<typename It>
	inline void Recv(It first, It last, const Datatype &datatype, const int src, const int tag, const Comm &comm) {
		typedef typename std::iterator_traits<It>::iterator_category category;
		MEL::Recv(first, last, datatype, src, tag, comm, category());
	};
};